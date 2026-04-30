"""
全局统计构建器

生成object_statistics_global.json文件
汇总所有场景的对象统计信息
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict


class GlobalStatisticsBuilder:
    """
    全局统计构建器
    
    汇总多个场景的对象统计信息，生成全局统计文件
    """
    
    def __init__(self):
        """初始化构建器"""
        pass
    
    def build(
        self,
        scene_metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建全局统计数据
        
        Args:
            scene_metadata_list: 场景元数据列表，每个元素包含完整的metadata信息
            
        Returns:
            全局统计数据字典，包含：
            - total_scenes: 场景总数
            - total_frames: 帧总数
            - per_scene_statistics: 每个场景的统计摘要
            - global_summary: 全局汇总统计
        """
        total_scenes = len(scene_metadata_list)
        total_frames = 0
        per_scene_statistics = []
        
        # 用于汇总全局统计的数据结构
        global_visibility = defaultdict(lambda: defaultdict(lambda: {'count': 0}))
        global_distance = defaultdict(lambda: defaultdict(int))
        global_dynamic_static = defaultdict(lambda: {'dynamic': 0, 'static': 0})
        global_total_objects = 0
        
        # 遍历每个场景的元数据
        for metadata in scene_metadata_list:
            scene_name = metadata.get('scene_name', 'unknown')
            scene_token = metadata.get('scene_token', '')
            frame_count = metadata.get('frame_count', 0)
            object_stats = metadata.get('object_statistics', {})
            
            total_frames += frame_count
            
            # 提取场景级别的统计信息
            scene_stat = {
                'scene_name': scene_name,
                'scene_token': scene_token,
                'frame_count': frame_count,
                'total_objects': object_stats.get('total_unique_objects', 0),
                'visibility_distribution': object_stats.get('visibility_distribution', {}),
                'distance_distribution': object_stats.get('distance_distribution', {}),
                'dynamic_static_distribution': object_stats.get('dynamic_static_distribution', {})
            }
            per_scene_statistics.append(scene_stat)
            
            # 累加到全局统计
            global_total_objects += object_stats.get('total_unique_objects', 0)
            
            # 汇总可见性分布
            visibility_dist = object_stats.get('visibility_distribution', {})
            for category, levels in visibility_dist.items():
                for level_key, level_data in levels.items():
                    global_visibility[category][level_key]['count'] += level_data.get('count', 0)
            
            # 汇总距离分布
            distance_dist = object_stats.get('distance_distribution', {})
            for category, bins in distance_dist.items():
                for bin_name, count in bins.items():
                    global_distance[category][bin_name] += count
            
            # 汇总动静态分布
            dynamic_static_dist = object_stats.get('dynamic_static_distribution', {})
            for category, status_data in dynamic_static_dist.items():
                global_dynamic_static[category]['dynamic'] += status_data.get('dynamic', {}).get('count', 0)
                global_dynamic_static[category]['static'] += status_data.get('static', {}).get('count', 0)
        
        # 计算全局百分比
        global_visibility_with_percentage = self._calculate_visibility_percentages(global_visibility)
        global_dynamic_static_with_percentage = self._calculate_dynamic_static_percentages(global_dynamic_static)
        
        # 构建全局汇总
        global_summary = {
            'total_objects_all_scenes': global_total_objects,
            'visibility_distribution': global_visibility_with_percentage,
            'distance_distribution': dict(global_distance),
            'dynamic_static_distribution': global_dynamic_static_with_percentage
        }
        
        # 构建完整的全局统计数据
        global_statistics = {
            'total_scenes': total_scenes,
            'total_frames': total_frames,
            'per_scene_statistics': per_scene_statistics,
            'global_summary': global_summary
        }
        
        return global_statistics
    
    def _calculate_visibility_percentages(
        self,
        visibility_data: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        计算可见性分布的百分比
        
        Args:
            visibility_data: 可见性统计数据
            
        Returns:
            包含count和percentage的可见性分布
        """
        result = {}
        
        for category, levels in visibility_data.items():
            # 计算该类别的总对象数
            total_count = sum(level_data['count'] for level_data in levels.values())
            
            category_dist = {}
            for level_key, level_data in levels.items():
                count = level_data['count']
                percentage = (count / total_count * 100.0) if total_count > 0 else 0.0
                category_dist[level_key] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
            
            result[category] = category_dist
        
        return result
    
    def _calculate_dynamic_static_percentages(
        self,
        dynamic_static_data: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        计算动静态分布的百分比
        
        Args:
            dynamic_static_data: 动静态统计数据
            
        Returns:
            包含count和percentage的动静态分布
        """
        result = {}
        
        for category, status_counts in dynamic_static_data.items():
            dynamic_count = status_counts['dynamic']
            static_count = status_counts['static']
            total_count = dynamic_count + static_count
            
            dynamic_percentage = (dynamic_count / total_count * 100.0) if total_count > 0 else 0.0
            static_percentage = (static_count / total_count * 100.0) if total_count > 0 else 0.0
            
            result[category] = {
                'dynamic': {
                    'count': dynamic_count,
                    'percentage': round(dynamic_percentage, 2)
                },
                'static': {
                    'count': static_count,
                    'percentage': round(static_percentage, 2)
                }
            }
        
        return result
    
    def save(
        self,
        global_statistics: Dict[str, Any],
        output_dir: Path,
        filename: str = 'object_statistics_global.json'
    ) -> Path:
        """
        保存全局统计数据到JSON文件
        
        Args:
            global_statistics: 全局统计数据字典
            output_dir: 输出目录
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(global_statistics, f, indent=2, ensure_ascii=False)
        
        return file_path

