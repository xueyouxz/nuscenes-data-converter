"""
静态地图构建器

生成static_map.bin文件（MessagePack格式）
包含车道线、道路边界、人行横道等
"""

from typing import Dict, Any, Union
from pathlib import Path

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline


class MapBuilder(BaseBuilder):
    """
    静态地图构建器
    
    构建场景的静态地图文件（static_map.bin）
    """
    
    def __init__(self, nuscenes_extractor=None, map_extractor=None):
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
        构建静态地图
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            地图数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            # 从pipeline获取缓存的地图
            static_map = scene_token_or_pipeline.get_static_map().copy()
            static_map['scene_token'] = scene_token_or_pipeline.scene_token
        else:
            # 传统模式
            scene_token = scene_token_or_pipeline
            ego_poses = self.nusc_extractor.extract_ego_poses(scene_token)
            static_map = self.map_extractor.extract_static_map(scene_token, ego_poses)
            static_map['scene_token'] = scene_token
        
        return static_map
    
    def save(
        self,
        static_map: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存静态地图到MessagePack文件
        
        Args:
            static_map: 地图数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_msgpack(static_map, output_dir, scene_name, 'static_map.bin')

