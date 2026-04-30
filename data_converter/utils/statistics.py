"""
统计计算工具模块

提供对象统计、道路统计和指标聚合功能
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def compute_object_statistics(
    annotations_per_frame: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    计算对象统计信息
    
    Args:
        annotations_per_frame: 每帧的标注列表
            格式: [[{category: "car", instance_id: 1, ...}, ...], ...]
        
    Returns:
        对象统计信息字典，包含：
        - objects_per_frame: 每帧对象数量数组
        - category_counts_per_frame: 每帧各类别数量
        - total_unique_objects: 唯一对象总数
        - total_by_category: 各类别对象总数
        
    原理：
        1. 一次遍历完成所有统计
        2. 使用instance_id去重，计算唯一对象数
    """
    frame_count = len(annotations_per_frame)
    
    # 初始化统计数据
    objects_per_frame = []
    category_counts_per_frame_dict = defaultdict(lambda: [0] * frame_count)
    unique_instance_ids = set()
    category_instance_sets = defaultdict(set)
    
    # 一次遍历完成所有统计
    for frame_idx, frame_annotations in enumerate(annotations_per_frame):
        objects_per_frame.append(len(frame_annotations))
        
        for ann in frame_annotations:
            category = ann.get('category', 'unknown')
            instance_id = ann.get('instance_id', None)
            
            # 统计当前帧类别数量
            category_counts_per_frame_dict[category][frame_idx] += 1
            
            # 收集唯一实例ID
            if instance_id is not None:
                unique_instance_ids.add(instance_id)
                category_instance_sets[category].add(instance_id)
    
    # 计算各类别总数（去重后）
    total_by_category = {
        category: len(instance_set)
        for category, instance_set in category_instance_sets.items()
    }
    
    return {
        'objects_per_frame': objects_per_frame,
        'category_counts_per_frame': dict(category_counts_per_frame_dict),
        'total_unique_objects': len(unique_instance_ids),
        'total_by_category': total_by_category
    }


def compute_road_statistics(
    static_map: Dict[str, List],
    annotations_per_frame: List[List[Dict[str, Any]]] = None
) -> Dict[str, int]:
    """
    计算道路统计信息
    
    Args:
        static_map: 静态地图数据，包含dividers, boundaries, ped_crossings等
        annotations_per_frame: 每帧标注（可选），用于统计交通设施
        
    Returns:
        道路统计信息字典，包含：
        - crosswalk_count: 人行横道数量
        - intersection_count: 交叉口数量（估算）
        - lane_count: 车道数量（估算）
        - traffic_light_count: 红绿灯数量
        - stop_sign_count: 停止标志数量
        
    原理：
        1. 直接统计地图元素数量
        2. 通过车道分隔线数量估算车道数
        3. 通过分隔线交叉点估算交叉口数量
        4. 从标注中统计交通设施（如果提供）
    """
    statistics = {
        'crosswalk_count': 0,
        'intersection_count': 0,
        'lane_count': 0,
        'traffic_light_count': 0,
        'stop_sign_count': 0
    }
    
    # 统计人行横道数量
    if 'ped_crossings' in static_map:
        statistics['crosswalk_count'] = len(static_map['ped_crossings'])
    elif 'ped_crossing' in static_map:
        statistics['crosswalk_count'] = len(static_map['ped_crossing'])
    
    # 统计车道数量（基于车道分隔线数量估算）
    if 'dividers' in static_map:
        divider_count = len(static_map['dividers'])
        # 估算：N条分隔线大约对应N+1条车道
        statistics['lane_count'] = max(divider_count + 1, 0)
    elif 'divider' in static_map:
        divider_count = len(static_map['divider'])
        statistics['lane_count'] = max(divider_count + 1, 0)
    
    # 估算交叉口数量（简化方法：基于车道分隔线的复杂度）
    # 如果有多条分隔线且存在人行横道，可能存在交叉口
    if statistics['lane_count'] >= 4 and statistics['crosswalk_count'] > 0:
        # 粗略估算：每2个人行横道对应1个交叉口
        statistics['intersection_count'] = max(statistics['crosswalk_count'] // 2, 1)
    
    # 统计交通设施（从标注中）
    if annotations_per_frame:
        traffic_lights = set()
        stop_signs = set()
        
        for frame_annotations in annotations_per_frame:
            for ann in frame_annotations:
                category = ann.get('category', '')
                instance_id = ann.get('instance_id', None)
                
                if 'traffic' in category.lower() or 'light' in category.lower():
                    if instance_id:
                        traffic_lights.add(instance_id)
                elif 'stop' in category.lower() or 'sign' in category.lower():
                    if instance_id:
                        stop_signs.add(instance_id)
        
        statistics['traffic_light_count'] = len(traffic_lights)
        statistics['stop_sign_count'] = len(stop_signs)
    
    return statistics


def aggregate_metrics(
    frame_metrics: List[Dict[str, Any]],
    metric_keys: List[str] = None
) -> Dict[str, float]:
    """
    聚合评估指标
    
    Args:
        frame_metrics: 每帧的指标列表
            格式: [{"mAP": 0.5, "NDS": 0.6, ...}, ...]
        metric_keys: 需要聚合的指标键列表，None表示所有数值指标
        
    Returns:
        聚合后的指标字典，只包含平均值（简化版）
        
    原理：
        简化聚合逻辑，只返回平均值，减少冗余信息
    """
    if not frame_metrics:
        return {}
    
    # 收集所有数值型指标
    metric_values = defaultdict(list)
    
    for frame_metric in frame_metrics:
        if not isinstance(frame_metric, dict):
            continue
        
        for key, value in frame_metric.items():
            # 跳过非数值类型
            if not isinstance(value, (int, float)):
                continue
            
            # 如果指定了metric_keys，只处理指定的指标
            if metric_keys is not None and key not in metric_keys:
                continue
            
            metric_values[key].append(value)
    
    # 计算平均值（简化版本，不再返回max/min/std）
    aggregated = {}
    for key, values in metric_values.items():
        if values:
            aggregated[key] = float(np.mean(values))
    
    return aggregated


def compute_per_category_statistics(
    annotations: List[Dict[str, Any]],
    categories: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    计算每个类别的统计信息
    
    Args:
        annotations: 标注列表
        categories: 类别列表
        
    Returns:
        每个类别的统计信息
        
    原理：
        为每个类别统计对象数量、平均尺寸、速度分布等
    """
    category_stats = {category: {
        'count': 0,
        'avg_size': [0.0, 0.0, 0.0],  # [w, l, h]
        'avg_velocity': 0.0,
        'size_std': [0.0, 0.0, 0.0]
    } for category in categories}
    
    # 按类别分组
    category_objects = defaultdict(list)
    for ann in annotations:
        category = ann.get('category', 'unknown')
        if category in categories:
            category_objects[category].append(ann)
    
    # 计算每个类别的统计
    for category, objects in category_objects.items():
        if not objects:
            continue
        
        category_stats[category]['count'] = len(objects)
        
        # 收集尺寸和速度数据
        sizes = []
        velocities = []
        
        for obj in objects:
            if 'size' in obj:
                sizes.append(obj['size'])
            if 'velocity' in obj:
                vel = obj['velocity']
                if isinstance(vel, (list, tuple)) and len(vel) >= 2:
                    velocities.append(np.linalg.norm(vel[:2]))
        
        # 计算平均尺寸
        if sizes:
            sizes_array = np.array(sizes)
            category_stats[category]['avg_size'] = sizes_array.mean(axis=0).tolist()
            category_stats[category]['size_std'] = sizes_array.std(axis=0).tolist()
        
        # 计算平均速度
        if velocities:
            category_stats[category]['avg_velocity'] = float(np.mean(velocities))
    
    return category_stats


def compute_visibility_distribution(
    annotations_per_frame: List[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    计算可见性分布统计
    
    Args:
        annotations_per_frame: 每帧的标注列表
            格式: [[{category: "car", instance_id: 1, visibility_level: 2, ...}, ...], ...]
        
    Returns:
        按类别和可见性等级的对象分布字典
        格式: {
            "pedestrian": {
                "level_0": {"count": 10, "percentage": 5.2},
                "level_1": {"count": 30, "percentage": 15.6},
                ...
            },
            ...
        }
        
    原理：
        1. 遍历所有帧的annotations，按category分组
        2. 使用instance_id去重，统计每个visibility_level的唯一对象数量
        3. 计算每个等级的百分比
    """
    # 为每个类别和可见性等级收集唯一的instance_id
    # category -> visibility_level -> set of instance_ids
    category_visibility_instances = defaultdict(lambda: defaultdict(set))
    
    for frame_annotations in annotations_per_frame:
        for ann in frame_annotations:
            category = ann.get('category', 'unknown')
            instance_id = ann.get('instance_id', None)
            visibility_level = ann.get('visibility_level', 0)
            
            if instance_id is not None:
                # 使用字符串格式化可见性等级
                level_key = f"level_{visibility_level}"
                category_visibility_instances[category][level_key].add(instance_id)
    
    # 计算统计数据
    visibility_distribution = {}
    
    for category, levels_dict in category_visibility_instances.items():
        # 计算该类别的总对象数（所有等级的并集）
        all_instances = set()
        for instance_set in levels_dict.values():
            all_instances.update(instance_set)
        
        total_count = len(all_instances)
        
        # 为每个可见性等级计算count和percentage
        category_dist = {}
        for level_key in ['level_0', 'level_1', 'level_2', 'level_3', 'level_4']:
            count = len(levels_dict.get(level_key, set()))
            percentage = (count / total_count * 100.0) if total_count > 0 else 0.0
            category_dist[level_key] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        visibility_distribution[category] = category_dist
    
    return visibility_distribution


def compute_distance_distribution(
    annotations_per_frame: List[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, int]]:
    """
    计算距离分布统计
    
    Args:
        annotations_per_frame: 每帧的标注列表
            格式: [[{category: "car", instance_id: 1, distance_to_ego: 15.5, ...}, ...], ...]
        
    Returns:
        按类别和距离区间的对象分布字典
        格式: {
            "pedestrian": {
                "0-10m": 45,
                "10-30m": 89,
                "30-50m": 32,
                "50+m": 26
            },
            ...
        }
        
    原理：
        1. 定义距离区间 [0-10, 10-30, 30-50, 50+]
        2. 统计每个instance在其首次出现时的距离所属区间
        3. 使用instance_id去重
    """
    # 距离区间定义
    distance_bins = [
        ('0-10m', 0, 10),
        ('10-30m', 10, 30),
        ('30-50m', 30, 50),
        ('50+m', 50, float('inf'))
    ]
    
    # 记录每个instance首次出现的距离
    # category -> bin_name -> set of instance_ids
    category_distance_instances = defaultdict(lambda: defaultdict(set))
    instance_first_distance = {}  # instance_id -> (category, distance)
    
    for frame_annotations in annotations_per_frame:
        for ann in frame_annotations:
            category = ann.get('category', 'unknown')
            instance_id = ann.get('instance_id', None)
            distance = ann.get('distance_to_ego', 0.0)
            
            if instance_id is not None:
                # 只记录首次出现的距离
                if instance_id not in instance_first_distance:
                    instance_first_distance[instance_id] = (category, distance)
    
    # 将instance分配到对应的距离区间
    for instance_id, (category, distance) in instance_first_distance.items():
        for bin_name, bin_min, bin_max in distance_bins:
            if bin_min <= distance < bin_max:
                category_distance_instances[category][bin_name].add(instance_id)
                break
    
    # 计算统计数据
    distance_distribution = {}
    
    for category, bins_dict in category_distance_instances.items():
        category_dist = {}
        for bin_name, _, _ in distance_bins:
            count = len(bins_dict.get(bin_name, set()))
            category_dist[bin_name] = count
        
        distance_distribution[category] = category_dist
    
    return distance_distribution


def compute_dynamic_static_distribution(
    annotations_per_frame: List[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    计算动态/静态对象分布统计
    
    Args:
        annotations_per_frame: 每帧的标注列表
            格式: [[{category: "car", instance_id: 1, is_moving: True, ...}, ...], ...]
        
    Returns:
        按类别的动态/静态对象分布字典
        格式: {
            "pedestrian": {
                "dynamic": {"count": 120, "percentage": 65.2},
                "static": {"count": 64, "percentage": 34.8}
            },
            ...
        }
        
    原理：
        1. 统计每个instance的is_moving状态
        2. 若instance在任意帧为True，则视为动态对象
        3. 计算动态和静态对象的数量和百分比
    """
    # category -> status -> set of instance_ids
    category_dynamic_instances = defaultdict(set)
    category_static_instances = defaultdict(set)
    instance_moving_status = {}  # instance_id -> (category, has_moved)
    
    # 遍历所有帧，收集每个instance的移动状态
    for frame_annotations in annotations_per_frame:
        for ann in frame_annotations:
            category = ann.get('category', 'unknown')
            instance_id = ann.get('instance_id', None)
            is_moving = ann.get('is_moving', False)
            
            if instance_id is not None:
                if instance_id not in instance_moving_status:
                    instance_moving_status[instance_id] = (category, False)
                
                # 如果该instance在任意帧移动过，标记为动态
                if is_moving:
                    current_category, _ = instance_moving_status[instance_id]
                    instance_moving_status[instance_id] = (current_category, True)
    
    # 分类为动态或静态
    for instance_id, (category, has_moved) in instance_moving_status.items():
        if has_moved:
            category_dynamic_instances[category].add(instance_id)
        else:
            category_static_instances[category].add(instance_id)
    
    # 计算统计数据
    dynamic_static_distribution = {}
    
    # 获取所有出现过的类别
    all_categories = set(category_dynamic_instances.keys()) | set(category_static_instances.keys())
    
    for category in all_categories:
        dynamic_count = len(category_dynamic_instances.get(category, set()))
        static_count = len(category_static_instances.get(category, set()))
        total_count = dynamic_count + static_count
        
        dynamic_percentage = (dynamic_count / total_count * 100.0) if total_count > 0 else 0.0
        static_percentage = (static_count / total_count * 100.0) if total_count > 0 else 0.0
        
        dynamic_static_distribution[category] = {
            'dynamic': {
                'count': dynamic_count,
                'percentage': round(dynamic_percentage, 2)
            },
            'static': {
                'count': static_count,
                'percentage': round(static_percentage, 2)
            }
        }
    
    return dynamic_static_distribution


def compute_enhanced_object_statistics(
    annotations_per_frame: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    计算增强的对象统计信息
    
    整合基础统计、可见性分布、距离分布和动静态分布
    
    Args:
        annotations_per_frame: 每帧的标注列表
        
    Returns:
        完整的对象统计信息字典，包含：
        - objects_per_frame: 每帧对象数量数组
        - category_counts_per_frame: 每帧各类别数量
        - total_unique_objects: 唯一对象总数
        - total_by_category: 各类别对象总数
        - visibility_distribution: 可见性分布
        - distance_distribution: 距离分布
        - dynamic_static_distribution: 动静态分布
    """
    # 计算基础统计
    basic_stats = compute_object_statistics(annotations_per_frame)
    
    # 计算增强统计
    visibility_dist = compute_visibility_distribution(annotations_per_frame)
    distance_dist = compute_distance_distribution(annotations_per_frame)
    dynamic_static_dist = compute_dynamic_static_distribution(annotations_per_frame)
    
    # 合并所有统计信息
    enhanced_stats = {
        **basic_stats,
        'visibility_distribution': visibility_dist,
        'distance_distribution': distance_dist,
        'dynamic_static_distribution': dynamic_static_dist
    }
    
    return enhanced_stats

