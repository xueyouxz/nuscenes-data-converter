"""
关联索引构建器（优化版本）

生成associations.json文件
包含GT与预测之间的双向查询索引

优化说明：
- 使用统一的匹配器（ObjectMatcher, MapElementMatcher）
- 复用pipeline中的缓存数据
- 消除重复的距离计算函数
"""

import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline
from ..utils.matching import ObjectMatcher


class AssociationsBuilder(BaseBuilder):
    """
    关联索引构建器
    
    构建场景的关联索引文件（associations.json）
    """
    
    def __init__(
        self,
        nuscenes_extractor=None,
        sparsedrive_extractor=None,
        map_extractor=None
    ):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器（传统模式）
            sparsedrive_extractor: SparseDrive预测提取器（传统模式）
            map_extractor: 地图提取器（传统模式）
        """
        self.nusc_extractor = nuscenes_extractor
        self.sd_extractor = sparsedrive_extractor
        self.map_extractor = map_extractor
        
        # 初始化匹配器
        self.object_matcher = ObjectMatcher()
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建关联索引
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            关联索引字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            return self._build_from_pipeline(scene_token_or_pipeline)
        else:
            return self._build_traditional(scene_token_or_pipeline)
    
    def _build_from_pipeline(self, pipeline: ScenePipeline) -> Dict[str, Any]:
        """
        从pipeline构建关联索引（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            
        Returns:
            关联索引字典
        """
        sample_tokens = pipeline.get_sample_tokens()
        
        # 构建对象关联
        object_associations, gt_to_pred_index, pred_to_gt_index = \
            self._build_object_associations_from_pipeline(pipeline, sample_tokens)
        
        # 构建地图关联
        map_associations, gt_to_pred_map_index, pred_to_gt_map_index = \
            self._build_map_associations_from_pipeline(pipeline, sample_tokens)
        
        associations = {
            'scene_token': pipeline.scene_token,
            'object_associations': object_associations,
            'indexes': {
                'gt_to_pred_objects': gt_to_pred_index,
                'pred_to_gt_objects': pred_to_gt_index,
                'gt_to_pred_map_elements': gt_to_pred_map_index,
                'pred_to_gt_map_elements': pred_to_gt_map_index
            },
            'map_associations': map_associations
        }
        
        return associations
    
    def _build_traditional(self, scene_token: str) -> Dict[str, Any]:
        """
        传统方式构建关联索引（向后兼容）
        
        Args:
            scene_token: 场景token
            
        Returns:
            关联索引字典
        """
        sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
        
        # 构建对象关联
        object_associations, gt_to_pred_index, pred_to_gt_index = \
            self._build_object_associations_traditional(sample_tokens)
        
        # 构建地图关联
        map_associations, gt_to_pred_map_index, pred_to_gt_map_index = \
            self._build_map_associations_traditional(sample_tokens, scene_token)
        
        associations = {
            'scene_token': scene_token,
            'object_associations': object_associations,
            'indexes': {
                'gt_to_pred_objects': gt_to_pred_index,
                'pred_to_gt_objects': pred_to_gt_index,
                'gt_to_pred_map_elements': gt_to_pred_map_index,
                'pred_to_gt_map_elements': pred_to_gt_map_index
            },
            'map_associations': map_associations
        }
        
        return associations
    
    def _build_object_associations_from_pipeline(
        self,
        pipeline: ScenePipeline,
        sample_tokens: List[str]
    ) -> tuple:
        """
        从pipeline构建对象关联索引（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            sample_tokens: sample token列表
            
        Returns:
            (associations_list, gt_to_pred_index, pred_to_gt_index)
        """
        associations = []
        gt_to_pred_index = {}
        pred_to_gt_index = {}
        
        for frame_index, sample_token in enumerate(sample_tokens):
            if not pipeline.sd_extractor.has_prediction(sample_token):
                continue
            
            # 从pipeline获取缓存的数据
            gt_annotations = pipeline.get_gt_annotations(frame_index, sample_token)
            pred_detections = pipeline.get_pred_detections(frame_index, sample_token)
            
            if not gt_annotations or not pred_detections:
                continue
            
            # 从pipeline获取缓存的匹配结果
            matches = pipeline.get_object_matches(frame_index, sample_token)
            matched_pred_indices = {pred_idx for _, pred_idx, _, _ in matches}
            
            # 处理TP对象
            for gt_idx, pred_idx, iou, distance in matches:
                gt_ann = gt_annotations[gt_idx]
                pred_det = pred_detections[pred_idx]
                
                gt_instance_id = gt_ann['instance_id']
                pred_instance_id = pred_det['instance_id']
                
                # 计算详细误差指标
                errors = pipeline.object_matcher.compute_object_errors(
                    gt_ann, pred_det, distance
                )
                
                # 添加到关联列表
                associations.append({
                    'frame_index': frame_index,
                    'pred_instance_id': pred_instance_id,
                    'gt_instance_id': gt_instance_id,
                    'category': gt_ann['category'],
                    'score': round(float(pred_det.get('score', 0)), 6),
                    'iou': round(float(iou), 6),
                    'distance': round(float(distance), 6),
                    'is_tp': True,
                    'errors': errors
                })
                
                # 更新索引
                if gt_instance_id not in gt_to_pred_index:
                    gt_to_pred_index[gt_instance_id] = {}
                if frame_index not in gt_to_pred_index[gt_instance_id]:
                    gt_to_pred_index[gt_instance_id][frame_index] = []
                gt_to_pred_index[gt_instance_id][frame_index].append(pred_instance_id)
                
                if pred_instance_id not in pred_to_gt_index:
                    pred_to_gt_index[pred_instance_id] = {}
                pred_to_gt_index[pred_instance_id][frame_index] = gt_instance_id
            
            # 处理FP对象
            for pred_idx, pred_det in enumerate(pred_detections):
                if pred_idx not in matched_pred_indices:
                    associations.append({
                        'frame_index': frame_index,
                        'pred_instance_id': pred_det['instance_id'],
                        'category': pred_det.get('category', 'unknown'),
                        'score': round(float(pred_det.get('score', 0)), 6),
                        'is_tp': False
                    })
        
        return associations, gt_to_pred_index, pred_to_gt_index
    
    def _build_object_associations_traditional(
        self,
        sample_tokens: List[str]
    ) -> tuple:
        """
        传统方式构建对象关联索引
        
        Args:
            sample_tokens: sample token列表
            
        Returns:
            (associations_list, gt_to_pred_index, pred_to_gt_index)
        """
        associations = []
        gt_to_pred_index = {}
        pred_to_gt_index = {}
        
        for frame_index, sample_token in enumerate(sample_tokens):
            if not self.sd_extractor.has_prediction(sample_token):
                continue
            
            gt_annotations = self.nusc_extractor.extract_annotations(sample_token)
            pred_detections = self.sd_extractor.extract_detections(sample_token)
            
            if not gt_annotations or not pred_detections:
                continue
            
            # 使用统一的匹配器
            matches = self.object_matcher.match(gt_annotations, pred_detections)
            matched_pred_indices = {pred_idx for _, pred_idx, _, _ in matches}
            
            # 处理TP对象
            for gt_idx, pred_idx, iou, distance in matches:
                gt_ann = gt_annotations[gt_idx]
                pred_det = pred_detections[pred_idx]
                
                gt_instance_id = gt_ann['instance_id']
                pred_instance_id = pred_det['instance_id']
                
                # 计算详细误差指标
                errors = self.object_matcher.compute_object_errors(
                    gt_ann, pred_det, distance
                )
                
                associations.append({
                    'frame_index': frame_index,
                    'pred_instance_id': pred_instance_id,
                    'gt_instance_id': gt_instance_id,
                    'category': gt_ann['category'],
                    'score': round(float(pred_det.get('score', 0)), 6),
                    'iou': round(float(iou), 6),
                    'distance': round(float(distance), 6),
                    'is_tp': True,
                    'errors': errors
                })
                
                if gt_instance_id not in gt_to_pred_index:
                    gt_to_pred_index[gt_instance_id] = {}
                if frame_index not in gt_to_pred_index[gt_instance_id]:
                    gt_to_pred_index[gt_instance_id][frame_index] = []
                gt_to_pred_index[gt_instance_id][frame_index].append(pred_instance_id)
                
                if pred_instance_id not in pred_to_gt_index:
                    pred_to_gt_index[pred_instance_id] = {}
                pred_to_gt_index[pred_instance_id][frame_index] = gt_instance_id
            
            # 处理FP对象
            for pred_idx, pred_det in enumerate(pred_detections):
                if pred_idx not in matched_pred_indices:
                    associations.append({
                        'frame_index': frame_index,
                        'pred_instance_id': pred_det['instance_id'],
                        'category': pred_det.get('category', 'unknown'),
                        'score': round(float(pred_det.get('score', 0)), 6),
                        'is_tp': False
                    })
        
        return associations, gt_to_pred_index, pred_to_gt_index
    
    def _build_map_associations_from_pipeline(
        self,
        pipeline: ScenePipeline,
        sample_tokens: List[str]
    ) -> tuple:
        """
        从pipeline构建地图关联索引（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            sample_tokens: sample token列表
            
        Returns:
            (associations_list, gt_to_pred_index, pred_to_gt_index)
        """
        associations = []
        gt_to_pred_index = {}
        pred_to_gt_index = {}
        
        # 遍历每一帧进行匹配
        for frame_index, sample_token in enumerate(sample_tokens):
            if not pipeline.sd_extractor.has_prediction(sample_token):
                continue
            
            pred_map = pipeline.get_pred_map(frame_index, sample_token)
            if not pred_map:
                continue
            
            # 获取所有类别的匹配结果（复用pipeline缓存）
            all_matches = pipeline.get_all_map_matches(frame_index, sample_token)
            
            # 处理每个类别的匹配
            for class_name, matches in all_matches.items():
                for pred_idx, gt_idx, chamfer_dist in matches:
                    pred_item = pred_map[pred_idx]
                    pred_score = pred_item.get('score', 0.0)
                    
                    gt_element_id = f"{class_name}:{gt_idx}"
                    
                    associations.append({
                        'frame_index': frame_index,
                        'pred_index': pred_idx,
                        'gt_element_id': gt_element_id,
                        'category': class_name,
                        'score': round(float(pred_score), 6),
                        'chamfer_distance': round(float(chamfer_dist), 6),
                        'min_distance': round(float(chamfer_dist), 6)
                    })
                    
                    # 更新索引
                    if gt_element_id not in gt_to_pred_index:
                        gt_to_pred_index[gt_element_id] = {}
                    if frame_index not in gt_to_pred_index[gt_element_id]:
                        gt_to_pred_index[gt_element_id][frame_index] = []
                    gt_to_pred_index[gt_element_id][frame_index].append(pred_idx)
                    
                    pred_key = f"{frame_index}:{class_name}:{pred_idx}"
                    pred_to_gt_index[pred_key] = gt_element_id
        
        return associations, gt_to_pred_index, pred_to_gt_index
    
    def _build_map_associations_traditional(
        self,
        sample_tokens: List[str],
        scene_token: str
    ) -> tuple:
        """
        传统方式构建地图关联索引
        
        Args:
            sample_tokens: sample token列表
            scene_token: 场景token
            
        Returns:
            (associations_list, gt_to_pred_index, pred_to_gt_index)
        """
        associations = []
        gt_to_pred_index = {}
        pred_to_gt_index = {}
        
        if not sample_tokens:
            return associations, gt_to_pred_index, pred_to_gt_index
        
        # 获取GT地图
        ego_poses = []
        for sample_token in sample_tokens:
            ego_pose = self.nusc_extractor.extract_ego_pose(sample_token)
            if ego_pose:
                ego_poses.append(ego_pose)
        
        if not ego_poses:
            return associations, gt_to_pred_index, pred_to_gt_index
        
        if self.map_extractor is None:
            from ..core.map_extractor import MapExtractor
            map_extractor = MapExtractor(self.nusc_extractor.nusc)
        else:
            map_extractor = self.map_extractor
        
        gt_map = map_extractor.extract_static_map(scene_token, ego_poses)
        
        # 遍历每一帧进行匹配
        for frame_index, sample_token in enumerate(sample_tokens):
            if not self.sd_extractor.has_prediction(sample_token):
                continue
            
            pred_map = self.sd_extractor.extract_map_predictions(sample_token)
            if not pred_map:
                continue
            
            # 使用统一的地图匹配器
            from ..utils.matching import MapElementMatcher
            map_matcher = MapElementMatcher()
            
            # 按类别匹配
            for class_name in ["ped_crossing", "divider", "boundary"]:
                matches = map_matcher.match(pred_map, gt_map, class_name)
                
                for pred_idx, gt_idx, chamfer_dist in matches:
                    pred_item = pred_map[pred_idx]
                    pred_score = pred_item.get('score', 0.0)
                    
                    gt_element_id = f"{class_name}:{gt_idx}"
                    
                    associations.append({
                        'frame_index': frame_index,
                        'pred_index': pred_idx,
                        'gt_element_id': gt_element_id,
                        'category': class_name,
                        'score': round(float(pred_score), 6),
                        'chamfer_distance': round(float(chamfer_dist), 6),
                        'min_distance': round(float(chamfer_dist), 6)
                    })
                    
                    if gt_element_id not in gt_to_pred_index:
                        gt_to_pred_index[gt_element_id] = {}
                    if frame_index not in gt_to_pred_index[gt_element_id]:
                        gt_to_pred_index[gt_element_id][frame_index] = []
                    gt_to_pred_index[gt_element_id][frame_index].append(pred_idx)
                    
                    pred_key = f"{frame_index}:{class_name}:{pred_idx}"
                    pred_to_gt_index[pred_key] = gt_element_id
        
        return associations, gt_to_pred_index, pred_to_gt_index
    
    def save(
        self,
        associations: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存关联索引到JSON文件
        
        Args:
            associations: 关联索引数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_json(associations, output_dir, scene_name, 'associations.json')
