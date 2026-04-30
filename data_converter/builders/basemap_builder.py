"""
栅格地图构建器

生成basemap.png和basemap_metadata.json文件
包含场景的栅格底图及坐标转换参数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
from pathlib import Path
from datetime import datetime
from PIL import Image

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline
from ..config import config


class BasemapBuilder(BaseBuilder):
    """
    栅格地图构建器
    
    构建场景的栅格底图文件（basemap.png）和元数据文件（basemap_metadata.json）
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
        构建栅格底图
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            包含图像数组和元数据的字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            return self._build_from_pipeline(scene_token_or_pipeline)
        else:
            return self._build_traditional(scene_token_or_pipeline)
    
    def _build_from_pipeline(self, pipeline: ScenePipeline) -> Dict[str, Any]:
        """从pipeline构建栅格底图"""
        # 获取场景信息
        scene_info = pipeline.nusc_extractor.extract_scene_info(pipeline.scene_token)
        scene_token = pipeline.scene_token
        scene_name = scene_info['scene_name']
        
        # 获取场景的地图范围
        ego_poses = pipeline.nusc_extractor.extract_ego_poses(scene_token)
        map_range = self.map_extractor._compute_map_range(
            ego_poses, 
            config.MAP_MARGIN
        )
        
        # 获取地图探索器
        map_explorer = self.map_extractor._get_map_explorer(scene_token)
        
        # 渲染栅格图像
        image_array, bounds = self._render_basemap(map_explorer, map_range)
        
        # 生成元数据
        metadata = self._generate_metadata(
            scene_token, 
            scene_name, 
            image_array, 
            bounds
        )
        
        return {
            'image': image_array,
            'metadata': metadata
        }
    
    def _build_traditional(self, scene_token: str) -> Dict[str, Any]:
        """传统方式构建栅格底图"""
        # 获取场景信息
        scene_info = self.nusc_extractor.extract_scene_info(scene_token)
        scene_name = scene_info['scene_name']
        
        # 获取场景的地图范围
        ego_poses = self.nusc_extractor.extract_ego_poses(scene_token)
        map_range = self.map_extractor._compute_map_range(
            ego_poses, 
            config.MAP_MARGIN
        )
        
        # 获取地图探索器
        map_explorer = self.map_extractor._get_map_explorer(scene_token)
        
        # 渲染栅格图像
        image_array, bounds = self._render_basemap(map_explorer, map_range)
        
        # 生成元数据
        metadata = self._generate_metadata(
            scene_token, 
            scene_name, 
            image_array, 
            bounds
        )
        
        return {
            'image': image_array,
            'metadata': metadata
        }
    
    def _render_basemap(self, map_explorer, map_range):
        """
        渲染栅格底图
        
        Args:
            map_explorer: NuScenesMapExplorer实例
            map_range: 场景的地图范围（Polygon）
            
        Returns:
            (image_array, bounds) - 图像数组和边界
        """
        # 获取地图范围的边界
        minx, miny, maxx, maxy = map_range.bounds
        width_meters = maxx - minx
        height_meters = maxy - miny
        
        # 计算图像尺寸（基于配置的分辨率）
        pixels_per_meter = config.BASEMAP_PIXELS_PER_METER
        width_pixels = int(width_meters * pixels_per_meter)
        height_pixels = int(height_meters * pixels_per_meter)
        
        # 创建patch_box (center_x, center_y, height, width)
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        patch_box = (center_x, center_y, height_meters, width_meters)
        
        # 计算figsize（英寸）
        dpi = config.BASEMAP_DPI
        figsize_width = width_pixels / dpi
        figsize_height = height_pixels / dpi
        
        # 渲染地图
        fig, ax = map_explorer.render_map_patch(
            patch_box,
            config.BASEMAP_LAYERS,
            figsize=(max(figsize_width, 1), max(figsize_height, 1)),
            alpha=0.5,
            render_egoposes_range=False,
            render_legend=False
        )
        
        # 移除坐标轴和边距
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # 转换为图像数组
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # 返回图像和边界
        bounds = {
            'min_x': float(minx),
            'max_x': float(maxx),
            'min_y': float(miny),
            'max_y': float(maxy),
            'width_meters': float(width_meters),
            'height_meters': float(height_meters)
        }
        
        return image_array, bounds
    
    def _generate_metadata(
        self, 
        scene_token: str, 
        scene_name: str, 
        image_array: np.ndarray, 
        bounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        生成栅格底图的元数据
        
        Args:
            scene_token: 场景token
            scene_name: 场景名称
            image_array: 图像数组
            bounds: 地图边界
            
        Returns:
            元数据字典
        """
        height, width = image_array.shape[:2]
        
        # 计算分辨率
        meters_per_pixel_x = bounds['width_meters'] / width
        meters_per_pixel_y = bounds['height_meters'] / height
        pixels_per_meter = 1.0 / meters_per_pixel_x  # 假设x和y相同
        
        metadata = {
            'scene_token': scene_token,
            'scene_name': scene_name,
            
            'image': {
                'filename': 'basemap.png',
                'width': int(width),
                'height': int(height),
                'format': 'PNG'
            },
            
            'coordinate_system': {
                'type': 'global',
                'unit': 'meters',
                'description': 'nuScenes全局坐标系，原点为数据集定义原点'
            },
            
            'bounds': bounds,
            
            'resolution': {
                'meters_per_pixel_x': float(meters_per_pixel_x),
                'meters_per_pixel_y': float(meters_per_pixel_y),
                'pixels_per_meter': float(pixels_per_meter)
            },
            
            'transform': {
                'origin_x': bounds['min_x'],
                'origin_y': bounds['max_y'],
                'y_axis_direction': 'down',
                'notes': '图像原点在左上角，Y轴向下；全局坐标Y轴向上'
            },
            
            'rendering': {
                'layers': config.BASEMAP_LAYERS,
                'dpi': config.BASEMAP_DPI,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        }
        
        return metadata
    
    def save(
        self,
        basemap_data: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Dict[str, Path]:
        """
        保存栅格底图和元数据
        
        Args:
            basemap_data: 包含image和metadata的字典
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径字典
        """
        scene_dir = self._get_scene_dir(output_dir, scene_name)
        
        # 保存PNG图片
        image_path = scene_dir / 'basemap.png'
        image_array = basemap_data['image']
        image = Image.fromarray(image_array)
        image.save(image_path, 'PNG')
        
        # 保存元数据JSON
        metadata_path = self.save_json(
            basemap_data['metadata'],
            output_dir,
            scene_name,
            'basemap_metadata.json'
        )
        
        return {
            'image': image_path,
            'metadata': metadata_path
        }


