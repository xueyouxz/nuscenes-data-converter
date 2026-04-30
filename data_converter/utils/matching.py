"""
匹配工具模块

集中实现对象匹配和地图元素匹配的通用逻辑
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.spatial.distance import cdist

from .distance import interpolate_line, chamfer_distance_batch
from ..config import config


class ObjectMatcher:
    """
    对象匹配器
    
    用于匹配GT对象和预测对象
    """
    
    def __init__(self, distance_threshold: float = None):
        """
        初始化对象匹配器
        
        Args:
            distance_threshold: 距离阈值（米），默认使用config中的值
        """
        self.distance_threshold = distance_threshold or config.OBJECT_MATCH_DISTANCE_THRESHOLD
    
    def match(
        self,
        gt_annotations: List[Dict],
        pred_detections: List[Dict]
    ) -> List[Tuple[int, int, float, float]]:
        """
        匹配GT和预测对象
        
        Args:
            gt_annotations: GT标注列表
            pred_detections: 预测检测列表
            
        Returns:
            匹配列表 [(gt_idx, pred_idx, iou, distance), ...]
            
        原理：
            1. 计算距离矩阵
            2. 贪心匹配：为每个GT找最近的预测
            3. 使用距离阈值过滤
            4. 计算简化的IoU（基于尺寸）
        """
        if not gt_annotations or not pred_detections:
            return []
        
        # 计算距离矩阵
        gt_centers = np.array([ann['translation'][:2] for ann in gt_annotations])
        pred_centers = np.array([det['translation'][:2] for det in pred_detections])
        
        distance_matrix = cdist(gt_centers, pred_centers)
        
        # 贪心匹配（优化版本）
        matches = self._greedy_match(
            distance_matrix, gt_annotations, pred_detections
        )
        
        return matches
    
    def _greedy_match(
        self,
        distance_matrix: np.ndarray,
        gt_annotations: List[Dict],
        pred_detections: List[Dict]
    ) -> List[Tuple[int, int, float, float]]:
        """
        贪心匹配算法（优化版本）
        
        Args:
            distance_matrix: 距离矩阵
            gt_annotations: GT标注列表
            pred_detections: 预测检测列表
            
        Returns:
            匹配列表
        """
        matches = []
        matched_pred_indices = set()
        
        for gt_idx in range(len(gt_annotations)):
            # 使用向量化操作找到最近的未匹配预测
            distances = distance_matrix[gt_idx].copy()
            
            # 将已匹配的预测的距离设为无穷大
            for matched_idx in matched_pred_indices:
                distances[matched_idx] = np.inf
            
            # 找到最小距离和对应的索引
            min_distance = distances.min()
            best_pred_idx = distances.argmin()
            
            # 检查是否匹配成功
            if min_distance < self.distance_threshold:
                matched_pred_indices.add(best_pred_idx)
                
                # 计算简化的IoU
                iou = self._compute_iou(
                    gt_annotations[gt_idx], pred_detections[best_pred_idx]
                )
                
                matches.append((gt_idx, best_pred_idx, iou, float(min_distance)))
        
        return matches
    
    def _compute_iou(self, gt_ann: Dict, pred_det: Dict) -> float:
        """
        计算简化的IoU（基于3D尺寸）
        
        Args:
            gt_ann: GT标注
            pred_det: 预测检测
            
        Returns:
            IoU值
        """
        gt_size = np.array(gt_ann['size'])
        pred_size = np.array(pred_det['size'])
        
        intersection = np.minimum(gt_size, pred_size).prod()
        union = gt_size.prod() + pred_size.prod() - intersection
        iou = intersection / (union + 1e-10)
        
        return float(iou)
    
    def compute_object_errors(
        self,
        gt_ann: Dict,
        pred_det: Dict,
        distance: float
    ) -> Dict[str, Any]:
        """
        计算对象的详细误差指标
        
        Args:
            gt_ann: GT标注
            pred_det: 预测检测
            distance: 中心距离（translation_error）
            
        Returns:
            误差指标字典
        """
        errors = {
            'translation_error': round(float(distance), 6)
        }
        
        # 尺寸误差（简化的IoU）
        gt_size = np.array(gt_ann.get('size', [1, 1, 1]))
        pred_size = np.array(pred_det.get('size', [1, 1, 1]))
        intersection = np.minimum(gt_size, pred_size).prod()
        union = gt_size.prod() + pred_size.prod() - intersection
        scale_error = 1 - intersection / (union + 1e-10)
        errors['scale_error'] = round(float(scale_error), 6)
        
        # 朝向误差
        pred_yaw = pred_det.get('yaw', 0)
        gt_yaw = gt_ann.get('yaw', 0)
        orientation_error = self._compute_angle_difference(pred_yaw, gt_yaw)
        errors['orientation_error'] = round(float(orientation_error), 6)
        
        # 速度误差
        pred_vel = np.array(pred_det.get('velocity', [0.0, 0.0]))
        gt_vel = np.array(gt_ann.get('velocity', [0.0, 0.0]))
        velocity_error = np.linalg.norm(pred_vel - gt_vel)
        errors['velocity_error'] = round(float(velocity_error), 6)
        
        # 属性误差
        pred_attr = pred_det.get('attribute_name', '')
        gt_attr = gt_ann.get('attribute_name', '')
        if gt_attr != '':
            attribute_error = 0.0 if pred_attr == gt_attr else 1.0
            errors['attribute_error'] = round(float(attribute_error), 6)
        
        return errors
    
    def _compute_angle_difference(self, angle1: float, angle2: float) -> float:
        """
        计算角度差（归一化到[-π, π]）
        
        Args:
            angle1: 角度1（弧度）
            angle2: 角度2（弧度）
            
        Returns:
            角度差的绝对值
        """
        diff = angle1 - angle2
        
        # 归一化到[-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        return abs(diff)


class MapElementMatcher:
    """
    地图元素匹配器
    
    用于匹配GT地图元素和预测地图元素
    """
    
    def __init__(
        self,
        interp_num: int = 200,
        match_threshold: float = 9.0
    ):
        """
        初始化地图元素匹配器
        
        Args:
            interp_num: 插值点数
            match_threshold: 匹配距离阈值（米）
        """
        self.interp_num = interp_num
        self.match_threshold = match_threshold
    
    def match(
        self,
        pred_map: List[Dict],
        gt_map: Dict[str, List],
        class_name: str
    ) -> List[Tuple[int, int, float]]:
        """
        匹配地图元素
        
        Args:
            pred_map: 预测地图列表
            gt_map: GT地图字典
            class_name: 类别名称
            
        Returns:
            匹配列表 [(pred_idx, gt_idx, chamfer_distance), ...]
            
        原理：
            1. 提取该类别的预测和GT向量
            2. 插值到固定点数
            3. 计算Chamfer距离矩阵
            4. 按置信度排序进行贪心匹配
            5. 使用距离阈值过滤
        """
        # 提取预测向量
        pred_vectors, pred_scores, pred_indices = self._extract_pred_vectors(
            pred_map, class_name
        )
        
        # 提取GT向量
        gt_vectors = self._extract_gt_vectors(gt_map, class_name)
        
        if not pred_vectors or not gt_vectors:
            return []
        
        # 插值到固定点数
        pred_lines = np.array([
            interpolate_line(vec, self.interp_num) for vec in pred_vectors
        ])
        gt_lines = np.array([
            interpolate_line(vec, self.interp_num) for vec in gt_vectors
        ])
        
        # 计算距离矩阵（使用优化的批量计算）
        dist_matrix = chamfer_distance_batch(pred_lines, gt_lines)
        
        # 按置信度排序进行贪心匹配
        matches = self._greedy_match_by_score(
            dist_matrix, pred_scores, pred_indices
        )
        
        return matches
    
    def _extract_pred_vectors(
        self,
        pred_map: List[Dict],
        class_name: str
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        从预测地图中提取指定类别的向量
        
        Args:
            pred_map: 预测地图列表
            class_name: 类别名称
            
        Returns:
            (向量列表, 置信度列表, 原始索引列表)
        """
        pred_vectors = []
        pred_scores = []
        pred_indices = []
        
        for idx, item in enumerate(pred_map):
            if item.get("category") == class_name:
                vectors = item.get("vectors", [])
                score = item.get("score", 0.0)
                if vectors and len(vectors) > 0:
                    pred_vectors.append(np.array(vectors))
                    pred_scores.append(score)
                    pred_indices.append(idx)
        
        return pred_vectors, pred_scores, pred_indices
    
    def _extract_gt_vectors(
        self,
        gt_map: Dict[str, List],
        class_name: str
    ) -> List[np.ndarray]:
        """
        从GT地图中提取指定类别的向量
        
        Args:
            gt_map: GT地图字典
            class_name: 类别名称
            
        Returns:
            向量列表
        """
        gt_vectors = gt_map.get(class_name, [])
        gt_vectors = [np.array(vec) for vec in gt_vectors if len(vec) > 0]
        return gt_vectors
    
    def _greedy_match_by_score(
        self,
        dist_matrix: np.ndarray,
        pred_scores: List[float],
        pred_indices: List[int]
    ) -> List[Tuple[int, int, float]]:
        """
        按置信度排序进行贪心匹配
        
        Args:
            dist_matrix: 距离矩阵 (M, K)
            pred_scores: 预测置信度列表
            pred_indices: 预测原始索引列表
            
        Returns:
            匹配列表 [(pred_idx, gt_idx, distance), ...]
        """
        # 按置信度从高到低排序
        pred_scores_array = np.array(pred_scores)
        sorted_idx = np.argsort(-pred_scores_array)
        
        matches = []
        gt_matched = set()
        
        for pred_orig_idx in sorted_idx:
            pred_idx = pred_indices[pred_orig_idx]
            
            # 找到最近的未匹配GT
            distances = dist_matrix[pred_orig_idx].copy()
            
            # 将已匹配的GT距离设为无穷大
            for matched_gt_idx in gt_matched:
                distances[matched_gt_idx] = np.inf
            
            min_dist = distances.min()
            min_gt_idx = int(distances.argmin())
            
            # 检查是否匹配
            if min_dist <= self.match_threshold and min_gt_idx not in gt_matched:
                matches.append((pred_idx, min_gt_idx, float(min_dist)))
                gt_matched.add(min_gt_idx)
        
        return matches


