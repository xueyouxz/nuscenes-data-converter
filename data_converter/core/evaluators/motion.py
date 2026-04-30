"""
运动预测评估器

基于temp/services/evaluates/motion_evaluator.py重构
实现运动预测的ADE、FDE、MR等指标
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from .detection import DetectionEvaluator


class MotionEvaluator:
    """
    运动预测评估器
    
    计算运动预测指标：ADE、FDE、Miss Rate、EPA
    """
    
    def __init__(self):
        self.miss_threshold = 2.0  # 错失率阈值(米)
        self.dist_threshold = 2.0  # 匹配距离阈值(米)
        self.invalid_value = 1e10  # 无效值占位符（用于JSON序列化，替代Infinity）
    
    @staticmethod
    def _sanitize_for_json(value: float) -> float:
        """
        将Infinity和NaN转换为JSON兼容的值
        
        Args:
            value: 需要清理的数值
            
        Returns:
            JSON兼容的数值
        """
        if np.isinf(value):
            return 1e10  # 使用大数值替代Infinity
        if np.isnan(value):
            return 0.0  # 使用0替代NaN
        return value
    
    def evaluate_sample(
        self,
        pred_detections: List[Dict[str, Any]],
        gt_trajectories: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估单帧运动预测
        
        Args:
            pred_detections: 预测框列表（包含trajectories）
            gt_trajectories: GT轨迹字典
            
        Returns:
            评估指标字典
            
        原理：
            1. 从预测框中提取轨迹
            2. 与GT轨迹匹配
            3. 计算ADE、FDE、MR等指标
        """
        # 转换预测数据格式
        pred_data = self._convert_pred_format(pred_detections)
        
        # 计算指标
        metrics = self._compute_motion_metrics(pred_data, gt_trajectories)
        
        return metrics
    
    def _convert_pred_format(
        self,
        pred_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        转换预测格式
        
        Args:
            pred_detections: 预测框列表
            
        Returns:
            包含trajs_3d, trajs_scores, labels_3d的字典
        """
        trajs_3d = []
        trajs_scores = []
        labels_3d = []
        
        for pred in pred_detections:
            category = pred.get("category", "")
            trajectories = pred.get("trajectories", [])
            trajectory_scores = pred.get("trajectory_scores", [])

            if category in ["car","bus","truck","bicycle","motorcycle", "pedestrian"]:
                if trajectory_scores and trajectories:
                    # 选择最佳轨迹（最高分数）
                    best_idx = np.argmax(trajectory_scores)
                    best_traj = trajectories[best_idx]
                    
                    trajs_3d.append(np.array(best_traj))
                    trajs_scores.append(float(trajectory_scores[best_idx]))
                    labels_3d.append(category)
        
        return {
            "trajs_3d": trajs_3d,
            "trajs_scores": trajs_scores,
            "labels_3d": labels_3d
        }
    
    def _compute_motion_metrics(
        self,
        pred_data: Dict[str, Any],
        gt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算运动预测指标
        
        Args:
            pred_data: 预测数据
            gt_data: GT数据
            
        Returns:
            指标字典
        """
        pred_trajs = pred_data.get("trajs_3d", [])
        pred_scores = pred_data.get("trajs_scores", [])
        pred_labels = pred_data.get("labels_3d", [])
        
        gt_trajs = gt_data.get("trajectories", [])
        gt_masks = gt_data.get("masks", [])
        gt_labels = gt_data.get("labels", [])
        
        # 按类别分别计算
        car_metrics = self._compute_class_metrics(
            pred_trajs, pred_scores, pred_labels,
            gt_trajs, gt_masks, gt_labels,
            "car"
        )
        ped_metrics = self._compute_class_metrics(
            pred_trajs, pred_scores, pred_labels,
            gt_trajs, gt_masks, gt_labels,
            "pedestrian"
        )
        
        # 计算平均指标（处理无效值）
        car_ade = car_metrics["min_ade"]
        ped_ade = ped_metrics["min_ade"]
        car_fde = car_metrics["min_fde"]
        ped_fde = ped_metrics["min_fde"]
        
        # 如果两个值都是无效值，平均值也是无效值；否则只计算有效值的平均
        if car_ade >= self.invalid_value and ped_ade >= self.invalid_value:
            avg_ade = self.invalid_value
        elif car_ade >= self.invalid_value:
            avg_ade = ped_ade
        elif ped_ade >= self.invalid_value:
            avg_ade = car_ade
        else:
            avg_ade = (car_ade + ped_ade) / 2
        
        if car_fde >= self.invalid_value and ped_fde >= self.invalid_value:
            avg_fde = self.invalid_value
        elif car_fde >= self.invalid_value:
            avg_fde = ped_fde
        elif ped_fde >= self.invalid_value:
            avg_fde = car_fde
        else:
            avg_fde = (car_fde + ped_fde) / 2
        
        avg_mr = (car_metrics["miss_rate"] + ped_metrics["miss_rate"]) / 2
        avg_epa = (car_metrics["epa"] + ped_metrics["epa"]) / 2
        
        # 优化：使用结构化数据组织类别指标，并清理JSON不兼容的值
        return {
            "minADE": self._sanitize_for_json(float(avg_ade)),
            "minFDE": self._sanitize_for_json(float(avg_fde)),
            "MR": float(avg_mr),
            "EPA": float(avg_epa),
            "metrics_by_class": {
                "car": {
                    "ADE": self._sanitize_for_json(float(car_metrics["min_ade"])),
                    "FDE": self._sanitize_for_json(float(car_metrics["min_fde"])),
                    "MR": float(car_metrics["miss_rate"]),
                    "EPA": float(car_metrics["epa"])
                },
                "pedestrian": {
                    "ADE": self._sanitize_for_json(float(ped_metrics["min_ade"])),
                    "FDE": self._sanitize_for_json(float(ped_metrics["min_fde"])),
                    "MR": float(ped_metrics["miss_rate"]),
                    "EPA": float(ped_metrics["epa"])
                }
            }
        }
    
    def _compute_class_metrics(
        self,
        pred_trajs: List,
        pred_scores: List,
        pred_labels: List,
        gt_trajs: List,
        gt_masks: List,
        gt_labels: List,
        class_name: str
    ) -> Dict[str, float]:
        """
        计算特定类别的指标
        
        Args:
            pred_trajs: 预测轨迹列表
            pred_scores: 预测分数列表
            pred_labels: 预测标签列表
            gt_trajs: GT轨迹列表
            gt_masks: GT掩码列表
            gt_labels: GT标签列表
            class_name: 类别名称
            
        Returns:
            指标字典
        """
        # 过滤指定类别
        pred_indices = [i for i, label in enumerate(pred_labels) if label == class_name]
        gt_indices = [i for i, label in enumerate(gt_labels) if label == class_name]
        
        if not pred_indices or not gt_indices:
            return {
                "min_ade": self.invalid_value,
                "min_fde": self.invalid_value,
                "miss_rate": 1.0,
                "epa": 0.0
            }
        
        # 提取对应类别的数据
        class_pred_trajs = [pred_trajs[i] for i in pred_indices]
        class_pred_scores = [pred_scores[i] for i in pred_indices]
        class_gt_trajs = [gt_trajs[i] for i in gt_indices]
        class_gt_masks = [gt_masks[i] for i in gt_indices]
        
        # 执行匹配和指标计算
        match_results = self._accumulate_matches(
            class_pred_trajs, class_pred_scores,
            class_gt_trajs, class_gt_masks
        )
        
        return self._compute_final_metrics(match_results)
    
    def _accumulate_matches(
        self,
        pred_trajs: List,
        pred_scores: List,
        gt_trajs: List,
        gt_masks: List
    ) -> Dict[str, List]:
        """
        贪心匹配算法
        
        Args:
            pred_trajs: 预测轨迹
            pred_scores: 预测分数
            gt_trajs: GT轨迹
            gt_masks: GT掩码
            
        Returns:
            匹配结果字典
            
        原理：
            使用 detection 的通用匹配函数，基于轨迹最小距离进行匹配
        """
        # 定义距离计算函数（基于轨迹最小距离）
        def distance_fn(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
            return self._compute_initial_distance(gt_traj, pred_traj)
        
        # 使用 detection 的通用匹配函数
        matches = DetectionEvaluator.greedy_match(
            pred_trajs, gt_trajs, pred_scores, distance_fn, threshold=self.dist_threshold
        )
        
        match_data = {
            'min_ade': [],
            'min_fde': [],
            'miss_rate': [],
            'epa_hits': [],
            'epa_fps': []
        }
        
        # 处理匹配结果
        for pred_idx, gt_idx, distance in matches:
            pred_traj = pred_trajs[pred_idx]
            
            if gt_idx is not None:
                # 匹配成功
                gt_traj_match = gt_trajs[gt_idx]
                gt_mask_match = gt_masks[gt_idx]
                
                # 计算轨迹预测指标
                minade, minfde, mr, epa_hit = self._prediction_metrics(
                    gt_traj_match, gt_mask_match, pred_traj
                )
                match_data['min_ade'].append(minade)
                match_data['min_fde'].append(minfde)
                match_data['miss_rate'].append(mr)
                match_data['epa_hits'].append(epa_hit)
            else:
                # 未匹配的预测作为假阳性
                match_data['epa_fps'].append(1)
        
        return match_data
    
    def _prediction_metrics(
        self,
        gt_traj: np.ndarray,
        gt_mask: np.ndarray,
        pred_traj: np.ndarray
    ) -> Tuple[float, float, float, int]:
        """
        计算单个预测的轨迹指标
        
        Args:
            gt_traj: GT轨迹
            gt_mask: GT掩码
            pred_traj: 预测轨迹
            
        Returns:
            (minADE, minFDE, miss_rate, epa_hit)
        """
        # 确保轨迹长度一致
        valid_steps = min(len(gt_traj), len(pred_traj))
        
        # 处理空轨迹的情况
        if valid_steps == 0:
            return self.invalid_value, self.invalid_value, 1.0, 0
        
        gt_traj_valid = gt_traj[:valid_steps]
        pred_traj_valid = pred_traj[:valid_steps]
        
        # 计算每个时间步的L2距离
        dist = np.linalg.norm(pred_traj_valid - gt_traj_valid, axis=1)
        
        # 计算关键指标
        minade = float(np.mean(dist))  # 平均位移误差
        minfde = float(dist[-1])       # 最终位移误差
        mr = float(dist.max() > self.miss_threshold)  # 错失率
        epa_hit = int(minfde < self.miss_threshold)   # EPA命中
        
        return minade, minfde, mr, epa_hit
    
    def _compute_initial_distance(
        self,
        gt_traj: np.ndarray,
        pred_traj: np.ndarray
    ) -> float:
        """
        计算GT和预测轨迹之间的最小距离
        
        Args:
            gt_traj: GT轨迹 (T, 2)
            pred_traj: 预测轨迹 (T, 2)
            
        Returns:
            最小距离
        """
        if len(gt_traj) == 0 or len(pred_traj) == 0:
            return self.invalid_value
        
        # 计算所有轨迹点两两之间的L2距离
        dists = np.linalg.norm(gt_traj[:, None, :] - pred_traj[None, :, :], axis=2)
        min_distance = float(np.min(dists))
        return min_distance
    
    def _compute_final_metrics(
        self,
        match_results: Dict[str, List]
    ) -> Dict[str, float]:
        """
        计算最终的聚合指标
        
        Args:
            match_results: 匹配结果
            
        Returns:
            最终指标
        """
        min_ades = match_results['min_ade']
        min_fdes = match_results['min_fde']
        miss_rates = match_results['miss_rate']
        epa_hits = match_results['epa_hits']
        epa_fps = match_results['epa_fps']
        
        # 计算平均指标
        avg_min_ade = np.mean(min_ades) if min_ades else self.invalid_value
        avg_min_fde = np.mean(min_fdes) if min_fdes else self.invalid_value
        avg_miss_rate = np.mean(miss_rates) if miss_rates else 1.0
        
        # 优化：EPA计算，直接使用数量而不是列表
        n_hits = len(epa_hits)
        n_fps = len(epa_fps)
        n_pos = n_hits + n_fps
        epa = (sum(epa_hits) - 0.5 * n_fps) / max(n_pos, 1) if n_pos > 0 else 0.0
        
        return {
            "min_ade": self._sanitize_for_json(float(avg_min_ade)),
            "min_fde": self._sanitize_for_json(float(avg_min_fde)),
            "miss_rate": float(avg_miss_rate),
            "epa": float(epa)
        }

