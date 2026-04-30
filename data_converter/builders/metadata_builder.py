"""
元数据构建器

生成metadata.json文件
包含场景信息、自车状态、对象统计、道路统计等
"""

from typing import Dict, Any, List, Union
from pathlib import Path

from .base_builder import BaseBuilder
from ..core.nuscenes_extractor import NuScenesExtractor
from ..core.pipeline import ScenePipeline
from ..utils.statistics import (
    compute_enhanced_object_statistics,
    compute_road_statistics
)


class MetadataBuilder(BaseBuilder):
    """
    元数据构建器
    
    构建场景的元数据文件（metadata.json）
    """
    
    def __init__(
        self,
        nuscenes_extractor: NuScenesExtractor = None,
        map_extractor=None
    ):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器
            map_extractor: 地图提取器
        """
        self.nusc_extractor = nuscenes_extractor
        self.map_extractor = map_extractor
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建元数据
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            元数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            return self._build_from_pipeline(scene_token_or_pipeline)
        else:
            return self._build_traditional(scene_token_or_pipeline)
    
    def _build_from_pipeline(self, pipeline: ScenePipeline) -> Dict[str, Any]:
        """从pipeline构建元数据"""
        scene_info = pipeline.nusc_extractor.extract_scene_info(pipeline.scene_token)
        sample_tokens = pipeline.get_sample_tokens()
        frame_count = len(sample_tokens)
        
        # 构建自车状态
        ego_states = self._build_ego_states(sample_tokens, pipeline.nusc_extractor)
        
        # 获取所有帧的标注
        annotations_per_frame = []
        for frame_index, sample_token in enumerate(sample_tokens):
            annotations = pipeline.get_gt_annotations(frame_index, sample_token)
            annotations_per_frame.append(annotations)
        
        # 计算增强的对象统计（包含可见性、距离、动静态分布）
        object_stats = compute_enhanced_object_statistics(annotations_per_frame)
        
        # 获取地图并计算道路统计
        static_map = pipeline.get_static_map()
        road_stats = compute_road_statistics(static_map, annotations_per_frame)
        
        return {
            'scene_token': pipeline.scene_token,
            'scene_name': scene_info['scene_name'],
            'scene_description': scene_info['scene_description'],
            'frame_count': frame_count,
            'ego_states': ego_states,
            'object_statistics': object_stats,
            'road_statistics': road_stats
        }
    
    def _build_traditional(self, scene_token: str) -> Dict[str, Any]:
        """传统方式构建元数据"""
        scene_info = self.nusc_extractor.extract_scene_info(scene_token)
        sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
        frame_count = len(sample_tokens)
        
        # 构建自车状态数组
        ego_states = self._build_ego_states(sample_tokens)
        
        # 获取所有帧的标注
        annotations_per_frame = []
        for sample_token in sample_tokens:
            annotations = self.nusc_extractor.extract_annotations(sample_token)
            annotations_per_frame.append(annotations)
        
        # 计算增强的对象统计（包含可见性、距离、动静态分布）
        object_stats = compute_enhanced_object_statistics(annotations_per_frame)
        
        # 提取地图并计算道路统计
        ego_poses = self.nusc_extractor.extract_ego_poses(scene_token)
        static_map = self.map_extractor.extract_static_map(
            scene_token, ego_poses
        )
        road_stats = compute_road_statistics(static_map, annotations_per_frame)
        
        # 构建元数据
        metadata = {
            'scene_token': scene_token,
            'scene_name': scene_info['scene_name'],
            'scene_description': scene_info['scene_description'],
            'frame_count': frame_count,
            'ego_states': ego_states,
            'object_statistics': object_stats,
            'road_statistics': road_stats
        }
        
        return metadata
    
    def _build_ego_states(
        self,
        sample_tokens: List[str],
        nusc_extractor=None
    ) -> List[Dict[str, Any]]:
        """
        构建自车状态数组
        
        Args:
            sample_tokens: sample token列表
            
        Returns:
            自车状态列表
            
        原理：
            为每一帧提取：
            - frame_index: 帧索引
            - timestamp: 时间戳
            - pose: 位姿（translation, rotation）
            - state: 车辆状态（velocity, acceleration, steering_angle, yaw_rate）
        """
        ego_states = []
        extractor = nusc_extractor or self.nusc_extractor
        
        for frame_index, sample_token in enumerate(sample_tokens):
            # 提取位姿
            ego_pose = extractor.extract_ego_pose(sample_token)
            
            # 提取状态
            ego_state = extractor.extract_ego_state(sample_token)
            
            # 组合数据
            state_dict = {
                'frame_index': frame_index,
                'timestamp': ego_pose['timestamp'],
                'pose': {
                    'translation': ego_pose['translation'],
                    'rotation': ego_pose['rotation']
                },
                'state': {
                    'velocity': ego_state['velocity'],
                    'acceleration': ego_state['acceleration'],
                    'steering_angle': ego_state['steering_angle'],
                    'yaw_rate': ego_state['yaw_rate']
                }
            }
            
            ego_states.append(state_dict)
        
        return ego_states
    
    def save(
        self,
        metadata: Dict[str, Any],
        output_dir: Path,
        scene_name: str = None
    ) -> Path:
        """
        保存元数据到JSON文件
        
        Args:
            metadata: 元数据字典
            output_dir: 输出目录
            scene_name: 场景名称（用于创建子目录）
            
        Returns:
            保存的文件路径
        """
        if not scene_name:
            scene_name = metadata['scene_name']
        
        return self.save_json(metadata, output_dir, scene_name, 'metadata.json')

