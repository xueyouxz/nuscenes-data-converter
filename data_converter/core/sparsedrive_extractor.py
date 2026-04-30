"""
SparseDrive预测提取器

基于temp/services/sparsedrive_service.py重构
提取SparseDrive模型的预测结果
"""

import pickle
import numpy as np
import torch
from typing import Dict, Any, List
from pathlib import Path

from ..utils.coord_transform import (
    transform_to_global,
    transform_yaw_to_global,
    ensure_quaternion
)
from ..config import config


# SparseDrive类别定义
CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

# 地图元素类别定义
MAP_CLASSES = (
    'ped_crossing',  # 0
    'divider',       # 1
    'boundary',      # 2
)


class SparseDriveExtractor:
    """
    SparseDrive预测提取器
    
    提取模型预测的检测框、轨迹、地图元素和规划结果
    """
    
    def __init__(self, prediction_file: str, nuscenes_extractor):
        """
        初始化提取器
        
        Args:
            prediction_file: 预测结果pkl文件路径
            nuscenes_extractor: NuScenesExtractor实例，用于获取ego_pose
        """
        self.prediction_file = prediction_file
        self.nuscenes_extractor = nuscenes_extractor
        self.predictions = self._load_predictions()
        self.sample_token_to_prediction = self._build_index()
    
    def _load_predictions(self) -> List[Dict]:
        """
        加载预测结果pkl文件
        
        Returns:
            预测结果列表
            
        原理：
            强制CPU加载，避免CUDA问题
            临时修改torch的设备检测和序列化设置
        """
        # 临时禁用CUDA
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        # 临时修改设备恢复逻辑
        original_restore = torch.serialization.default_restore_location
        torch.serialization.default_restore_location = lambda storage, location: storage
        
        with open(self.prediction_file, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复原始设置
        torch.cuda.is_available = original_is_available
        torch.serialization.default_restore_location = original_restore
        
        return data
    
    def _build_index(self) -> Dict[str, Dict]:
        """
        构建sample_token到预测结果的索引
        
        Returns:
            {sample_token: prediction_dict}
            
        原理：
            遍历预测列表，以sample_token为键建立字典索引
        """
        index = {}
        for prediction in self.predictions:
            token = prediction['img_bbox']['token']
            index[token] = prediction['img_bbox']
        return index
    
    def has_prediction(self, sample_token: str) -> bool:
        """
        检查是否有该sample的预测
        
        Args:
            sample_token: sample token
            
        Returns:
            是否存在预测
        """
        return sample_token in self.sample_token_to_prediction
    
    def extract_detections(
        self,
        sample_token: str,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        提取检测框（全局坐标系）
        
        Args:
            sample_token: sample token
            score_threshold: 置信度阈值，None表示使用配置的默认值
            
        Returns:
            检测框列表，每个包含category, translation, yaw, size, velocity, trajectories等
            
        原理：
            1. 从预测字典提取3D框、分数、类别
            2. 过滤低分预测
            3. 从Ego坐标系转换到全局坐标系
            4. 提取多模态轨迹并转换坐标
        """
        if sample_token not in self.sample_token_to_prediction:
            return []
        
        if score_threshold is None:
            score_threshold = config.DETECTION_SCORE_THRESHOLD
        
        pred = self.sample_token_to_prediction[sample_token]
        
        # 提取检测数据
        boxes_3d = pred['boxes_3d'].detach().cpu().numpy().copy()
        scores_3d = pred['scores_3d'].detach().cpu().numpy().copy()
        labels_3d = pred['labels_3d'].detach().cpu().numpy().copy()
        instance_ids = pred['instance_ids'].detach().cpu().numpy().copy()
        trajs_3d = pred['trajs_3d'].detach().cpu().numpy().copy()
        trajs_score = pred['trajs_score'].detach().cpu().numpy().copy()
        
        # 过滤低分预测
        valid_indices = scores_3d >= score_threshold
        if not np.any(valid_indices):
            return []
        
        boxes_3d = boxes_3d[valid_indices]
        scores_3d = scores_3d[valid_indices]
        labels_3d = labels_3d[valid_indices]
        instance_ids = instance_ids[valid_indices]
        trajs_3d = trajs_3d[valid_indices]
        trajs_score = trajs_score[valid_indices]
        
        # 获取ego_pose用于坐标转换
        ego_pose = self.nuscenes_extractor.extract_ego_pose(sample_token)
        ego_translation = ego_pose['translation']
        ego_rotation = ensure_quaternion(ego_pose['rotation'])
        
        # 批量转换坐标
        translations_local = boxes_3d[:, :2]
        yaws_local = boxes_3d[:, 6]
        sizes = boxes_3d[:, 3:6]
        velocities = boxes_3d[:, 7:9]
        
        # 转换到全局坐标系
        from nuscenes.prediction.helper import convert_local_coords_to_global
        translations_global = convert_local_coords_to_global(
            translations_local, ego_translation, ego_rotation
        )
        yaws_global = yaws_local + ego_pose['yaw']
        
        # 构建结果
        detections = []
        for i in range(len(boxes_3d)):
            # 转换轨迹到全局坐标系
            obj_trajs = trajs_3d[i]  # (6, 12, 2)
            obj_trajs_global = []
            
            for traj_idx in range(obj_trajs.shape[0]):
                traj = obj_trajs[traj_idx]  # (12, 2)
                traj_global = convert_local_coords_to_global(
                    traj, ego_translation, ego_rotation
                )
                obj_trajs_global.append(traj_global.tolist())
            
            detection = {
                'category': CLASSES[labels_3d[i]],
                'score': float(scores_3d[i]),
                'instance_id': int(instance_ids[i]),
                'translation': translations_global[i].tolist(),
                'yaw': float(yaws_global[i]),
                'size': sizes[i].tolist(),
                'velocity': velocities[i].tolist(),
                'trajectories': obj_trajs_global,  # 6个模态，每个12步
                'trajectory_scores': trajs_score[i].tolist(),
                'attribute_name': ""
            }
            detections.append(detection)
        
        return detections
    
    def extract_planning(self, sample_token: str) -> List[List[float]]:
        """
        提取规划轨迹（全局坐标系）
        
        Args:
            sample_token: sample token
            
        Returns:
            规划轨迹点列表 [[x, y], ...]
            
        原理：
            1. 提取final_planning
            2. 翻转y轴（修正横向偏移）
            3. 添加原点（0, 0）作为起点
            4. 从Ego坐标系转换到全局坐标系
        """
        if sample_token not in self.sample_token_to_prediction:
            return []
        
        pred = self.sample_token_to_prediction[sample_token]
        
        # 提取规划轨迹
        fplanning = pred['final_planning'].detach().cpu().numpy().copy()
        
        # 翻转y轴（横向坐标）
        fplanning[:, 0] = -fplanning[:, 0]
        
        # 添加原点
        zero_point = np.array([[0, 0]])
        fplanning = np.vstack([zero_point, fplanning])
        
        # 转换到全局坐标系
        ego_pose = self.nuscenes_extractor.extract_ego_pose(sample_token)
        ego_translation = ego_pose['translation']
        ego_rotation = ensure_quaternion(ego_pose['rotation'])
        
        from nuscenes.prediction.helper import convert_local_coords_to_global
        fplanning_global = convert_local_coords_to_global(
            fplanning, ego_translation, ego_rotation
        )
        
        return fplanning_global.tolist()
    
    def extract_map_predictions(
        self,
        sample_token: str,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        提取地图预测（全局坐标系）
        
        Args:
            sample_token: sample token
            score_threshold: 置信度阈值，None表示使用配置的默认值
            
        Returns:
            地图元素列表，每个包含category, vectors, score
            
        原理：
            1. 提取vectors, scores, labels
            2. 过滤低分预测
            3. 从Ego坐标系转换到全局坐标系
        """
        if sample_token not in self.sample_token_to_prediction:
            return []
        
        if score_threshold is None:
            score_threshold = config.DETECTION_SCORE_THRESHOLD
        
        pred = self.sample_token_to_prediction[sample_token]
        
        # 提取地图数据
        vectors = pred['vectors']
        scores = pred['scores']
        labels = pred['labels']
        
        # 转换为numpy数组
        if hasattr(scores, 'detach'):
            scores = scores.detach().cpu().numpy().copy()
        if hasattr(labels, 'detach'):
            labels = labels.detach().cpu().numpy().copy()
        
        # 过滤低分预测
        valid_indices = scores >= score_threshold
        if not np.any(valid_indices):
            return []
        
        # 过滤vectors
        if isinstance(vectors, list):
            valid_idx_list = np.where(valid_indices)[0].tolist()
            vectors = [vectors[i] for i in valid_idx_list]
        else:
            if hasattr(vectors, 'detach'):
                vectors = vectors.detach().cpu().numpy().copy()
            vectors = vectors[valid_indices]
        
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        
        # 获取ego_pose
        ego_pose = self.nuscenes_extractor.extract_ego_pose(sample_token)
        ego_translation = ego_pose['translation']
        ego_rotation = ensure_quaternion(ego_pose['rotation'])
        
        # 转换坐标
        from nuscenes.prediction.helper import convert_local_coords_to_global
        
        map_predictions = []
        for index, category_vectors in enumerate(vectors):
            label_id = labels[index]
            category_name = MAP_CLASSES[label_id]
            
            # 确保是numpy数组
            if hasattr(category_vectors, 'detach'):
                category_vectors = category_vectors.detach().cpu().numpy().copy()
            elif not isinstance(category_vectors, np.ndarray):
                category_vectors = np.array(category_vectors)
            
            # 转换到全局坐标系
            global_vectors = convert_local_coords_to_global(
                category_vectors, ego_translation, ego_rotation
            )
            
            map_predictions.append({
                'category': category_name,
                'score': float(scores[index]),
                'vectors': global_vectors.tolist()
            })
        
        return map_predictions

