"""
地图评估器（优化版本）

基于temp/services/evaluates/mapping_evaluator.py重构
实现地图分割的AP指标计算

优化说明：
- 使用通用的距离计算模块
- 消除重复的插值和chamfer distance函数
"""

import numpy as np
from typing import Dict, List, Any, Tuple

from ...utils.distance import interpolate_line, chamfer_distance_batch

# 地图类别
MAPPING_CLASSES = ["ped_crossing", "divider", "boundary"]

# 评估常量
INTERP_NUM = 200  # 向量插值固定点数
MAP_THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 9]  # 距离阈值(米)


class MappingEvaluator:
    """
    地图评估器
    
    计算地图预测的AP指标和Chamfer距离
    """

    def evaluate_sample(
            self,
            pred_map: List[Dict[str, Any]],
            gt_map: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        评估单帧地图预测
        
        Args:
            pred_map: 预测地图元素列表
            gt_map: GT地图字典
            
        Returns:
            评估指标字典，包含：
            - AP_by_class: 各类别平均AP
            - AP_by_threshold: 各阈值下平均AP
            - AP_by_class_threshold: 各类别在各阈值下的AP
            - mAP: 总体平均精度
            - prediction_errors: 预测误差（每个预测只记录一次）
        """
        if not pred_map:
            return self._empty_result()

        # 计算各类别的AP
        class_results = {}
        all_errors = []

        for class_name in MAPPING_CLASSES:
            class_result = self._compute_class_ap(
                pred_map, gt_map, class_name
            )
            class_results[class_name] = class_result
            # 合并误差信息（每个预测只记录一次）
            all_errors.extend(class_result["errors"])

        # 构建返回结果
        result = self._build_result(class_results)
        result["prediction_errors"] = all_errors

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果结构"""
        return {
            "AP_by_class": {
                "ped_crossing": 0.0,
                "divider": 0.0,
                "boundary": 0.0
            },
            "AP_by_threshold": {str(t): 0.0 for t in MAP_THRESHOLDS},
            "AP_by_class_threshold": {
                cls: {str(t): 0.0 for t in MAP_THRESHOLDS}
                for cls in MAPPING_CLASSES
            },
            "mAP": 0.0,
            "prediction_errors": []
        }

    def _build_result(
            self,
            class_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建评估结果
        
        Args:
            class_results: 各类别的评估结果
            
        Returns:
            结构化的评估结果字典
        """
        # 按类别汇总AP
        AP_by_class = {
            cls: class_results.get(cls, {}).get("avg_ap", 0.0)
            for cls in MAPPING_CLASSES
        }

        # 按阈值汇总AP（跨类别平均）
        AP_by_threshold = {}
        for threshold in MAP_THRESHOLDS:
            threshold_str = str(threshold)
            ap_list = [
                class_results.get(cls, {}).get("AP_by_threshold", {}).get(threshold_str, 0.0)
                for cls in MAPPING_CLASSES
            ]
            AP_by_threshold[threshold_str] = np.mean(ap_list) if ap_list else 0.0

        # 按类别和阈值组织AP
        AP_by_class_threshold = {
            cls: class_results.get(cls, {}).get("AP_by_threshold", {str(t): 0.0 for t in MAP_THRESHOLDS})
            for cls in MAPPING_CLASSES
        }

        # 计算总体mAP
        mAP = np.mean(list(AP_by_class.values())) if AP_by_class else 0.0

        return {
            "AP_by_class": AP_by_class,
            "AP_by_threshold": AP_by_threshold,
            "AP_by_class_threshold": AP_by_class_threshold,
            "mAP": float(mAP)
        }

    def _compute_class_ap(
            self,
            pred_map: List[Dict],
            gt_map: Dict[str, List],
            class_name: str
    ) -> Dict[str, Any]:
        """
        计算单个类别的AP
        
        Args:
            pred_map: 预测地图
            gt_map: GT地图
            class_name: 类别名称
            
        Returns:
            包含以下字段的字典：
            - avg_ap: 平均AP
            - AP_by_threshold: 各阈值下的AP
            - errors: 预测误差列表（每个预测只记录一次）
            - stats_by_threshold: 各阈值下的统计信息（TP/FP数量等）
        """
        # 提取预测向量
        pred_vectors = []
        pred_scores = []

        for item in pred_map:
            if item.get("category") == class_name:
                vectors = item.get("vectors", [])
                score = item.get("score", 0.0)
                if vectors:
                    pred_vectors.append(np.array(vectors))
                    pred_scores.append(score)

        # 提取GT向量
        gt_vectors = gt_map.get(class_name, [])
        gt_vectors = [np.array(vec) for vec in gt_vectors if len(vec) > 0]

        if not pred_vectors or not gt_vectors:
            return {
                "avg_ap": 0.0,
                "AP_by_threshold": {str(t): 0.0 for t in MAP_THRESHOLDS},
                "errors": [],
                "stats_by_threshold": {str(t): {"tp": 0, "fp": 0, "num_gt": len(gt_vectors)} for t in MAP_THRESHOLDS}
            }

        # 插值到固定点数（使用通用模块）
        pred_lines = np.array([
            interpolate_line(vec, INTERP_NUM) for vec in pred_vectors
        ])
        gt_lines = np.array([
            interpolate_line(vec, INTERP_NUM) for vec in gt_vectors
        ])

        # 优化：距离矩阵只计算一次（使用通用模块的优化批量计算）
        dist_matrix = chamfer_distance_batch(pred_lines, gt_lines)

        # 计算每个预测的最小距离（用于误差记录，只计算一次）
        pred_scores_array = np.array(pred_scores)
        min_distances = dist_matrix.min(axis=1)  # 每个预测到最近GT的距离

        # 计算不同阈值下的AP
        AP_by_threshold = {}
        stats_by_threshold = {}

        for threshold in MAP_THRESHOLDS:
            threshold_str = str(threshold)

            # 实例匹配
            tp, fp = self._instance_match(
                dist_matrix, pred_scores_array, threshold
            )

            # 计算AP
            ap = self._calculate_ap(tp, fp, len(gt_vectors))
            AP_by_threshold[threshold_str] = float(ap)

            # 保存统计信息
            stats_by_threshold[threshold_str] = {
                "tp": int(tp.sum()),
                "fp": int(fp.sum()),
                "num_pred": len(pred_vectors),
                "num_gt": len(gt_vectors)
            }

        # 计算平均AP
        avg_ap = np.mean(list(AP_by_threshold.values())) if AP_by_threshold else 0.0

        # 优化：每个预测只记录一次误差信息（不按阈值重复）
        errors = [
            {
                "class_name": class_name,
                "prediction_index": i,
                "score": float(score),
                "min_distance": float(min_dist)  # 移除重复的chamfer_distance字段
            }
            for i, (score, min_dist) in enumerate(zip(pred_scores, min_distances))
        ]

        return {
            "avg_ap": float(avg_ap),
            "AP_by_threshold": AP_by_threshold,
            "errors": errors,
            "stats_by_threshold": stats_by_threshold
        }


    def _instance_match(
            self,
            dist_matrix: np.ndarray,
            scores: np.ndarray,
            threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        实例匹配（贪心算法）
        
        Args:
            dist_matrix: 距离矩阵
            scores: 预测置信度
            threshold: 距离阈值
            
        Returns:
            (tp数组, fp数组)
            
        原理：
            按置信度排序，依次为每个预测找最近的未匹配GT
            距离小于阈值则为TP，否则为FP
        """
        num_pred, num_gt = dist_matrix.shape

        # 按置信度降序排列
        sorted_idx = np.argsort(-scores)

        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        gt_matched = set()

        for i in sorted_idx:
            if num_gt == 0:
                fp[i] = 1
                continue

            # 找到最近的GT
            min_dist = dist_matrix[i].min()
            min_idx = dist_matrix[i].argmin()

            # 检查是否匹配
            if min_dist <= threshold and min_idx not in gt_matched:
                tp[i] = 1
                gt_matched.add(min_idx)
            else:
                fp[i] = 1

        return tp, fp

    def _calculate_ap(
            self,
            tp: np.ndarray,
            fp: np.ndarray,
            num_gt: int
    ) -> float:
        """
        计算AP（11点插值法）
        
        Args:
            tp: TP数组
            fp: FP数组
            num_gt: GT总数
            
        Returns:
            AP分数
        """
        if num_gt == 0:
            return 0.0

        # 累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # 计算精确率和召回率
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # 11点插值
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask])

        return ap / 11
