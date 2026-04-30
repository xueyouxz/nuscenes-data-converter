"""
检测评估器

基于temp/services/evaluates/detection_evaluator.py重构
实现目标检测的mAP、NDS等指标计算
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Callable


# 检测类别
DETECTION_CLASSES = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
]


class DetectionEvaluator:
    """
    检测评估器
    
    计算检测指标：mAP、NDS、mATE、mASE、mAOE、mAVE、mAAE
    """
    
    def evaluate_sample(
        self,
        pred_boxes: List[Dict[str, Any]],
        gt_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        评估单帧检测结果
        
        Args:
            pred_boxes: 预测框列表
            gt_boxes: GT框列表
            
        Returns:
            评估指标字典，包含：
            - NDS, mAP: 总体指标
            - mATE, mASE, mAOE, mAVE, mAAE: 误差指标
            - AP_by_class: 各类别AP
            - per_object_metrics: 每个对象的精简指标（仅TP记录详细误差）
            
        原理：
            1. 按类别分别计算指标
            2. 使用距离阈值匹配预测和GT
            3. 计算TP、FP、FN
            4. 计算各项误差指标
            5. 聚合得到mAP、NDS等
        """
        if not pred_boxes:
            return {
                "NDS": 0.0,
                "mAP": 0.0,
                "mATE": 1.0,
                "mASE": 1.0,
                "mAOE": 1.0,
                "mAVE": 1.0,
                "mAAE": 0.0,
                "AP_by_class": {cls: 0.0 for cls in DETECTION_CLASSES},
                "per_object_metrics": []
            }
        
        # 按类别计算指标
        class_metrics, per_object_metrics = self._compute_class_metrics(
            pred_boxes, gt_boxes
        )
        
        # 聚合指标
        result = self._aggregate_metrics(class_metrics, gt_boxes)
        result["per_object_metrics"] = per_object_metrics
        
        return result
    
    def _compute_class_metrics(
        self,
        pred_boxes: List[Dict],
        gt_boxes: List[Dict]
    ) -> Tuple[Dict, List]:
        """
        按类别计算指标
        
        Args:
            pred_boxes: 预测框
            gt_boxes: GT框
            
        Returns:
            (class_metrics, per_object_metrics)
        """
        class_metrics = {}
        per_object_metrics = []
        
        for class_name in DETECTION_CLASSES:
            # 过滤当前类别
            class_pred = [box for box in pred_boxes if box.get("category") == class_name]
            class_gt = [box for box in gt_boxes if box.get("category") == class_name]
            
            # 初始化指标
            metrics = {
                "tp": [],
                "fp": [],
                "scores": [],
                "translation_error": [],
                "scale_error": [],
                "orientation_error": [],
                "velocity_error": [],
                "attribute_error": []
            }
            
            # 匹配并计算指标
            self._match_predictions_to_gt(
                class_pred, class_gt, metrics, per_object_metrics
            )
            
            class_metrics[class_name] = metrics
        
        return class_metrics, per_object_metrics
    
    def _match_predictions_to_gt(
        self,
        pred_boxes: List[Dict],
        gt_boxes: List[Dict],
        metrics: Dict,
        per_object_metrics: List
    ):
        """
        匹配预测框和GT框
        
        Args:
            pred_boxes: 预测框
            gt_boxes: GT框
            metrics: 累积指标
            per_object_metrics: 每个对象的指标列表
            
        原理：
            贪心匹配算法：
            1. 按置信度排序预测框
            2. 依次为每个预测框找最近的未匹配GT
            3. 距离小于阈值则匹配成功（TP），否则为FP
        """
        if not pred_boxes:
            return
        
        # 按置信度排序
        pred_boxes.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        if not gt_boxes:
            # 没有GT，所有预测都是FP
            for pred in pred_boxes:
                metrics["tp"].append(0)
                metrics["fp"].append(1)
                metrics["scores"].append(pred.get("score", 0))
                # 优化：FP只存储必要字段
                per_object_metrics.append({
                    "pred_instance_id": pred.get("instance_id"),
                    "category": pred.get("category"),
                    "score": pred.get("score", 0),
                    "is_tp": False
                })
            return
        
        # 贪心匹配
        self._greedy_matching(pred_boxes, gt_boxes, metrics, per_object_metrics)
    
    @staticmethod
    def greedy_match(
        pred_items: List[Any],
        gt_items: List[Any],
        pred_scores: List[float],
        distance_fn: Callable[[Any, Any], float],
        threshold: float = 2.0
    ) -> List[Tuple[int, int, float]]:
        """
        通用的贪心匹配算法
        
        Args:
            pred_items: 预测项列表
            gt_items: GT项列表
            pred_scores: 预测分数列表（用于排序）
            distance_fn: 距离计算函数，接受 (pred_item, gt_item) 返回距离
            threshold: 匹配距离阈值
            
        Returns:
            匹配结果列表，每个元素为 (pred_idx, gt_idx, distance)
            如果未匹配则为 (pred_idx, None, distance)
            
        原理：
            1. 按置信度排序预测项
            2. 依次为每个预测找最近的未匹配GT
            3. 距离小于阈值则匹配成功
        """
        if not pred_items:
            return []
        
        if not gt_items:
            # 没有GT，所有预测都未匹配
            return [(i, None, float('inf')) for i in range(len(pred_items))]
        
        # 按置信度排序，获取排序后的索引
        sorted_indices = sorted(
            range(len(pred_items)),
            key=lambda i: pred_scores[i] if i < len(pred_scores) else 0,
            reverse=True
        )
        
        matched_gt_indices = set()
        matches = []
        
        for pred_idx in sorted_indices:
            pred_item = pred_items[pred_idx]
            min_distance = np.inf
            best_gt_idx = None
            
            # 找到最近的未匹配GT
            for gt_idx in range(len(gt_items)):
                if gt_idx not in matched_gt_indices:
                    distance = distance_fn(pred_item, gt_items[gt_idx])
                    if distance < min_distance:
                        min_distance = distance
                        best_gt_idx = gt_idx
            
            # 判断是否匹配成功
            if best_gt_idx is not None and min_distance < threshold:
                matches.append((pred_idx, best_gt_idx, min_distance))
                matched_gt_indices.add(best_gt_idx)
            else:
                matches.append((pred_idx, None, min_distance))
        
        return matches
    
    def _greedy_matching(
        self,
        pred_boxes: List[Dict],
        gt_boxes: List[Dict],
        metrics: Dict,
        per_object_metrics: List
    ):
        """
        贪心匹配算法
        
        原理：
            使用通用匹配函数，基于2D中心点距离进行匹配
        """
        # 定义距离计算函数（基于2D中心点）
        def distance_fn(pred: Dict, gt: Dict) -> float:
            pred_center = np.array(pred["translation"][:2])
            gt_center = np.array(gt["translation"][:2])
            return float(np.linalg.norm(pred_center - gt_center))
        
        # 获取预测分数
        pred_scores = [box.get("score", 0) for box in pred_boxes]
        
        # 使用通用匹配函数
        matches = self.greedy_match(
            pred_boxes, gt_boxes, pred_scores, distance_fn, threshold=2.0
        )
        
        # 处理匹配结果
        for pred_idx, gt_idx, distance in matches:
            pred = pred_boxes[pred_idx]
            if gt_idx is not None:
                # TP
                self._record_match(
                    pred, gt_boxes[gt_idx], distance,
                    metrics, per_object_metrics
                )
            else:
                # FP
                self._record_false_positive(pred, metrics, per_object_metrics)
    
    def _record_match(
        self,
        pred: Dict,
        gt: Dict,
        distance: float,
        metrics: Dict,
        per_object_metrics: List
    ):
        """
        记录匹配成功（TP）
        
        计算各项误差：
        - translation_error: 中心距离
        - scale_error: 尺寸误差（1 - IoU）
        - orientation_error: 朝向误差
        - velocity_error: 速度误差
        - attribute_error: 属性分类误差
        """
        metrics["tp"].append(1)
        metrics["fp"].append(0)
        metrics["scores"].append(pred.get("score", 0))
        metrics["translation_error"].append(distance)
        
        # 计算尺寸误差（简化的IoU）
        pred_size = np.array(pred.get("size", [1, 1, 1]))
        gt_size = np.array(gt.get("size", [1, 1, 1]))
        
        intersection = np.minimum(pred_size, gt_size).prod()
        union = pred_size.prod() + gt_size.prod() - intersection
        scale_error = 1 - intersection / (union + 1e-10)
        metrics["scale_error"].append(scale_error)
        
        # 朝向误差
        pred_yaw = pred.get("yaw", 0)
        gt_yaw = gt.get("yaw", 0)
        orientation_error = self._compute_angle_difference(pred_yaw, gt_yaw)
        metrics["orientation_error"].append(orientation_error)
        
        # 速度误差
        pred_vel = np.array(pred.get("velocity", [0.0, 0.0]))
        gt_vel = np.array(gt.get("velocity", [0.0, 0.0]))
        velocity_error = np.linalg.norm(pred_vel - gt_vel)
        metrics["velocity_error"].append(velocity_error)
        
        # 属性误差
        pred_attr = pred.get("attribute_name", "")
        gt_attr = gt.get("attribute_name", "")
        if gt_attr == "":
            attribute_error = np.nan
        else:
            attribute_error = 0.0 if pred_attr == gt_attr else 1.0
        metrics["attribute_error"].append(attribute_error)
        
        # 记录每个对象的指标
        per_object_metrics.append({
            "pred_instance_id": pred.get("instance_id"),
            "gt_instance_id": gt.get("instance_id"),
            "category": pred.get("category"),
            "score": pred.get("score", 0),
            "is_tp": True,
            "translation_error": float(distance),
            "scale_error": float(scale_error),
            "orientation_error": float(orientation_error),
            "velocity_error": float(velocity_error),
            "attribute_error": float(attribute_error) if not np.isnan(attribute_error) else None
        })
    
    def _record_false_positive(
        self,
        pred: Dict,
        metrics: Dict,
        per_object_metrics: List
    ):
        """
        记录假阳性（FP）
        
        优化：只存储必要字段，不存储None值
        """
        metrics["tp"].append(0)
        metrics["fp"].append(1)
        metrics["scores"].append(pred.get("score", 0))
        
        # 优化：FP只存储必要字段，减少冗余
        per_object_metrics.append({
            "pred_instance_id": pred.get("instance_id"),
            "category": pred.get("category"),
            "score": pred.get("score", 0),
            "is_tp": False
        })
    
    def _aggregate_metrics(
        self,
        class_metrics: Dict,
        gt_boxes: List[Dict]
    ) -> Dict[str, float]:
        """
        聚合所有类别的指标
        
        计算：
        - mAP: 平均精度
        - mATE: 平均平移误差
        - mASE: 平均尺度误差
        - mAOE: 平均朝向误差
        - mAVE: 平均速度误差
        - mAAE: 平均属性误差
        - NDS: NuScenes检测分数
        - AP_by_class: 各类别AP（新增）
        """
        all_ap_scores = []
        all_translation_errors = []
        all_scale_errors = []
        all_orientation_errors = []
        all_velocity_errors = []
        all_attribute_errors = []
        AP_by_class = {}
        
        for class_name in DETECTION_CLASSES:
            metrics = class_metrics[class_name]
            
            if not metrics["tp"]:
                AP_by_class[class_name] = 0.0
                continue
            
            # 计算AP
            tp = np.array(metrics["tp"])
            fp = np.array(metrics["fp"])
            num_gt = len([box for box in gt_boxes if box.get("category") == class_name])
            
            ap_score = self._calculate_average_precision(tp, fp, num_gt)
            all_ap_scores.append(ap_score)
            AP_by_class[class_name] = float(ap_score)
            
            # 收集误差
            all_translation_errors.extend(metrics["translation_error"])
            all_scale_errors.extend(metrics["scale_error"])
            all_orientation_errors.extend(metrics["orientation_error"])
            all_velocity_errors.extend(metrics["velocity_error"])
            
            # 收集属性误差（排除nan）
            attr_errors = [e for e in metrics["attribute_error"] if not np.isnan(e)]
            all_attribute_errors.extend(attr_errors)
        
        # 计算平均指标
        mAP = np.mean(all_ap_scores) if all_ap_scores else 0.0
        mATE = np.mean(all_translation_errors) if all_translation_errors else 1.0
        mASE = np.mean(all_scale_errors) if all_scale_errors else 1.0
        mAOE = np.mean(all_orientation_errors) if all_orientation_errors else 1.0
        mAVE = np.mean(all_velocity_errors) if all_velocity_errors else 1.0
        mAAE = 1.0 - np.mean(all_attribute_errors) if all_attribute_errors else 1.0
        
        # 计算NDS
        nds = self._calculate_nds(mAP, mATE, mASE, mAOE, mAVE, mAAE)
        
        return {
            "NDS": float(nds),
            "mAP": float(mAP),
            "mATE": float(mATE),
            "mASE": float(mASE),
            "mAOE": float(mAOE),
            "mAVE": float(mAVE),
            "mAAE": float(mAAE),
            "AP_by_class": AP_by_class
        }
    
    def _calculate_average_precision(
        self,
        tp: np.ndarray,
        fp: np.ndarray,
        num_gt: int
    ) -> float:
        """
        计算平均精度（AP）
        
        原理：
            11点插值法
        """
        if num_gt == 0:
            return 0.0
        
        # 累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算召回率和精确率
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # 11点插值
        ap = 0.0
        for threshold in np.linspace(0, 1, 11):
            mask = recalls >= threshold
            if np.any(mask):
                ap += np.max(precisions[mask])
        
        return ap / 11
    
    def _calculate_nds(
        self,
        mAP: float,
        mATE: float,
        mASE: float,
        mAOE: float,
        mAVE: float,
        mAAE: float
    ) -> float:
        """
        计算NDS分数
        
        公式：NDS = (5 * mAP + sum(TP_scores)) / 10
        其中TP_score = max(1 - error, 0)
        """
        tp_errors = [
            max(0.0, 1 - mATE),
            max(0.0, 1 - mASE),
            max(0.0, 1 - mAOE / np.pi),
            max(0.0, 1 - mAVE),
            max(0.0, 1 - mAAE)
        ]
        
        nds = (5 * mAP + sum(tp_errors)) / 10.0
        return nds
    
    def _compute_angle_difference(self, angle1: float, angle2: float) -> float:
        """
        计算角度差（归一化到[-π, π]）
        """
        diff = angle1 - angle2
        
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        return abs(diff)

