"""
场景索引构建器

生成scenes_index.json文件
包含所有场景的基本信息、汇总指标和缩略图数据
"""

import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from .base_builder import BaseBuilder


class SceneIndexBuilder(BaseBuilder):
    """
    场景索引构建器
    
    构建所有场景的统一索引文件
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
    
    def build(self, scene_token_or_pipeline):
        """
        实现BaseBuilder的抽象方法
        
        SceneIndexBuilder不使用常规的build接口
        请使用build_index方法代替
        """
        raise NotImplementedError(
            "SceneIndexBuilder不支持build方法，请使用build_index(scene_tokens, output_dir)方法"
        )
    
    def build_index(
        self,
        scene_tokens: List[str],
        output_dir: Path,
        # sample_rate: int = 3
    ) -> Dict[str, Any]:
        """
        构建所有场景的索引
        
        Args:
            scene_tokens: 场景token列表
            output_dir: 输出目录（包含所有场景数据）
            sample_rate: 缩略图采样率（每N帧采样一次）
            
        Returns:
            场景索引数据
        """
        scenes = []
        
        for scene_token in scene_tokens:
            scene_info = self.nusc_extractor.extract_scene_info(scene_token)
            scene_name = scene_info['scene_name']
            scene_dir = output_dir  / scene_name
            
            metadata_file = scene_dir / 'metadata.json'
            metrics_file = scene_dir / 'metrics.json'
            # static_map_file = scene_dir / 'static_map.bin'
            
            metadata = self._load_json(metadata_file)
            metrics = self._load_json(metrics_file)
            
            # thumbnail_data = self._generate_thumbnail(
            #     metadata,
            #     static_map_file,
            #     sample_rate
            # )
            
            scene_item = {
                'scene_token': scene_token,
                'scene_name': scene_name,
                'scene_description': scene_info['scene_description'],
                'frame_count': metadata['frame_count'],
                'summary_metrics': metrics['scene_summary'],
                # 'thumbnail': thumbnail_data,
                # 'data_path': f'scenes/{scene_name}'
            }
            
            scenes.append(scene_item)
        
        return {
            'total_scenes': len(scenes),
            'scenes': scenes
        }
    
    def _generate_thumbnail(
        self,
        metadata: dict,
        static_map_file: Path,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        生成缩略图数据（简化版）
        
        Args:
            metadata: 场景元数据
            static_map_file: 静态地图文件路径
            sample_rate: 采样率
            
        Returns:
            缩略图数据
        """
        ego_states = metadata['ego_states']
        
        # 简化自车轨迹（每N帧采样）
        ego_trajectory = [
            [
                state['pose']['translation'][0],
                state['pose']['translation'][1]
            ]
            for i, state in enumerate(ego_states)
            if i % sample_rate == 0
        ]
        
        # 加载并简化地图
        static_map = self._load_msgpack(static_map_file)
        simplified_map = self._simplify_map(static_map)
        
        # 计算地图边界
        map_bounds = self._calculate_bounds(ego_trajectory, simplified_map)
        
        return {
            'ego_trajectory': ego_trajectory,
            'map_bounds': map_bounds,
            'simplified_map': simplified_map
        }
    
    def _simplify_map(self, static_map: dict) -> Dict[str, List]:
        """
        简化地图数据
        
        使用Douglas-Peucker算法简化折线
        
        Args:
            static_map: 完整的静态地图
            
        Returns:
            简化后的地图
        """
        tolerance = 2.0  # 简化容差（米）
        
        simplified = {
            'dividers': [],
            'boundaries': []
        }
        
        # 简化车道线
        for polyline in static_map.get('divider', [])[:10]:  # 只取前10条
            simplified_line = self._douglas_peucker(polyline, tolerance)
            simplified['dividers'].append(simplified_line)
        
        # 简化边界
        for polyline in static_map.get('boundary', [])[:5]:  # 只取前5条
            simplified_line = self._douglas_peucker(polyline, tolerance)
            simplified['boundaries'].append(simplified_line)
        
        return simplified
    
    def _douglas_peucker(
        self,
        points: List[List[float]],
        tolerance: float
    ) -> List[List[float]]:
        """
        Douglas-Peucker算法简化折线
        
        Args:
            points: 原始点序列
            tolerance: 容差
            
        Returns:
            简化后的点序列
        """
        if len(points) <= 2:
            return points
        
        # 找到距离起点-终点连线最远的点
        start = np.array(points[0])
        end = np.array(points[-1])
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            point = np.array(points[i])
            dist = self._point_to_line_distance(point, start, end)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # 如果最大距离大于容差，递归简化
        if max_dist > tolerance:
            left = self._douglas_peucker(points[:max_idx+1], tolerance)
            right = self._douglas_peucker(points[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]
    
    def _point_to_line_distance(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray
    ) -> float:
        """计算点到线段的距离"""
        line_vec = end - start
        point_vec = point - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        if proj_length < 0:
            return np.linalg.norm(point_vec)
        elif proj_length > line_len:
            return np.linalg.norm(point - end)
        else:
            proj_point = start + proj_length * line_unitvec
            return np.linalg.norm(point - proj_point)
    
    def _calculate_bounds(
        self,
        ego_trajectory: List[List[float]],
        simplified_map: dict
    ) -> Dict[str, float]:
        """
        计算地图边界
        
        Args:
            ego_trajectory: 自车轨迹
            simplified_map: 简化地图
            
        Returns:
            边界信息
        """
        all_points = ego_trajectory.copy()
        
        # 添加地图点
        for polylines in simplified_map.values():
            for polyline in polylines:
                all_points.extend(polyline)
        
        if not all_points:
            return {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
        
        all_points = np.array(all_points)
        
        return {
            'min_x': float(all_points[:, 0].min()),
            'min_y': float(all_points[:, 1].min()),
            'max_x': float(all_points[:, 0].max()),
            'max_y': float(all_points[:, 1].max())
        }
    
    def _load_json(self, file_path: Path) -> dict:
        """加载JSON文件"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _load_msgpack(self, file_path: Path) -> dict:
        """加载MessagePack文件"""
        import msgpack
        with open(file_path, 'rb') as f:
            return msgpack.unpackb(f.read(), raw=False)
    
    def save(
        self,
        index_data: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        保存场景索引到JSON文件
        
        Args:
            index_data: 索引数据
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / 'scenes_index.json'
        
        with open(file_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return file_path

