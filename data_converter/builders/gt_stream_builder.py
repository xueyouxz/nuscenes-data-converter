"""
GT流数据构建器

生成gt_stream.bin文件（MessagePack格式）
包含所有帧的GT对象数据
"""

from typing import Dict, Any, Union
from pathlib import Path

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline


class GtStreamBuilder(BaseBuilder):
    """
    GT流数据构建器
    
    构建场景的GT流文件（gt_stream.bin）
    """
    
    def __init__(self, nuscenes_extractor=None):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器
        """
        self.nusc_extractor = nuscenes_extractor
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建GT流数据
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            GT流数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            pipeline = scene_token_or_pipeline
            sample_tokens = pipeline.get_sample_tokens()
            scene_token = pipeline.scene_token
            nusc_extractor = pipeline.nusc_extractor
        else:
            scene_token = scene_token_or_pipeline
            sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
            nusc_extractor = self.nusc_extractor
        
        # 构建所有帧的数据
        frames = []
        for frame_index, sample_token in enumerate(sample_tokens):
            frame_data = self._build_frame(sample_token, frame_index, nusc_extractor)
            frames.append(frame_data)
        
        gt_stream = {
            'scene_token': scene_token,
            'frames': frames
        }
        
        return gt_stream
    
    def _build_frame(
        self,
        sample_token: str,
        frame_index: int,
        nusc_extractor=None
    ) -> Dict[str, Any]:
        """
        构建单帧GT数据
        
        Args:
            sample_token: sample token
            frame_index: 帧索引
            
        Returns:
            帧数据字典
            
        原理：
            提取：
            - ego_pose: 自车位姿
            - objects: 对象数据（instance_tokens, categories, boxes, velocities）
            所有坐标都在全局坐标系中
        """
        extractor = nusc_extractor or self.nusc_extractor
        
        # 提取ego pose
        ego_pose = extractor.extract_ego_pose(sample_token)
        
        # 提取标注（全局坐标系）
        annotations = extractor.extract_annotations(sample_token, 'global')
        
        # 组织对象数据
        instance_tokens = []
        instance_ids = []
        categories = []
        boxes = []
        velocities = []
        distances = []
        relative_velocities = []
        visibility_levels = []
        visibility_descriptions = []
        
        for ann in annotations:
            instance_tokens.append(ann['instance_token'])
            instance_ids.append(ann['instance_id'])
            categories.append(ann['category'])
            
            # 组装box：[x, y, z, w, l, h, yaw]
            trans = ann['translation']
            size = ann['size']
            yaw = ann['yaw']
            box = [
                trans[0], trans[1], trans[2],
                size[0], size[1], size[2],
                yaw
            ]
            boxes.append(box)
            
            velocities.append(ann['velocity'])
            
            # 添加新属性
            distances.append(ann.get('distance_to_ego', 0.0))
            relative_velocities.append(ann.get('relative_velocity', [0.0, 0.0]))
            visibility_levels.append(ann.get('visibility_level', 0))
            visibility_descriptions.append(ann.get('visibility_description', 'unknown'))
        
        frame_data = {
            'frame_index': frame_index,
            'timestamp': ego_pose['timestamp'],
            'ego_pose': {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            },
            'objects': {
                'instance_ids': instance_ids,
                'categories': categories,
                'boxes': boxes,
                'velocities': velocities,
                'distances_to_ego': distances,
                'relative_velocities': relative_velocities,
                'visibility_levels': visibility_levels,
                'visibility_descriptions': visibility_descriptions
            }
        }
        
        return frame_data
    
    def save(
        self,
        gt_stream: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存GT流到MessagePack文件
        
        Args:
            gt_stream: GT流数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_msgpack(gt_stream, output_dir, scene_name, 'gt_stream.bin')

