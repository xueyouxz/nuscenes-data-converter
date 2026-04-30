"""
NuScenes数据提取器

基于temp/services/nuscenes_service.py重构
提取NuScenes数据集的GT标注、自车状态等信息
"""

import numpy as np
from typing import List, Dict, Any, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from ..utils.coord_transform import (
    ensure_quaternion,
    transform_to_global,
    transform_yaw_to_global
)
from ..config import config


# 类别映射表：将NuScenes原始类别映射到简化类别
CATEGORY_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


class NuScenesExtractor:
    """
    NuScenes数据提取器
    
    封装所有NuScenes数据集的数据提取逻辑
    不涉及缓存，只负责数据提取
    """
    
    def __init__(self, dataroot: str, version: str = 'v1.0-trainval'):
        """
        初始化提取器
        
        Args:
            dataroot: NuScenes数据根目录
            version: 数据集版本
        """
        self.dataroot = dataroot
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.nusc_can = NuScenesCanBus(dataroot=dataroot)
    
    def extract_scene_info(self, scene_token: str) -> Dict[str, Any]:
        """
        提取场景基本信息
        
        Args:
            scene_token: 场景token
            
        Returns:
            场景信息字典，包含scene_token, scene_name, description等
            
        原理：
            从NuScenes数据库中查询scene记录，提取基本信息
        """
        scene = self.nusc.get('scene', scene_token)
        
        return {
            'scene_token': scene_token,
            'scene_name': scene['name'],
            'scene_description': scene['description'],
            'nbr_samples': scene['nbr_samples'],
            'first_sample_token': scene['first_sample_token'],
            'last_sample_token': scene['last_sample_token']
        }
    
    def get_sample_tokens(self, scene_token: str, max_samples: int = None) -> List[str]:
        """
        获取场景的所有sample token列表
        
        Args:
            scene_token: 场景token
            max_samples: 最大样本数，None表示不限制
            
        Returns:
            sample token列表
            
        原理：
            从first_sample开始，通过next链接遍历所有sample
        """
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        
        sample_tokens = []
        while sample_token:
            sample_tokens.append(sample_token)
            
            # 检查是否达到最大数量
            if max_samples and len(sample_tokens) >= max_samples:
                break
            
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
        
        return sample_tokens
    
    def extract_ego_pose(self, sample_token: str) -> Dict[str, Any]:
        """
        提取单帧的自车位姿
        
        Args:
            sample_token: sample token
            
        Returns:
            自车位姿字典，包含translation, rotation, yaw等
            
        原理：
            1. 通过sample获取LIDAR_TOP的sample_data
            2. 通过sample_data获取ego_pose
            3. 计算yaw角度
        """
        sample = self.nusc.get('sample', sample_token)
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        translation = ego_pose['translation']
        rotation = Quaternion(ego_pose['rotation'])
        yaw = quaternion_yaw(rotation)
        
        return {
            'sample_token': sample_token,
            'translation': translation,
            'rotation': rotation.q.tolist(),
            'yaw': float(yaw),
            'timestamp': sample['timestamp']
        }
    
    def extract_ego_poses(self, scene_token: str) -> List[Dict[str, Any]]:
        """
        提取场景所有帧的自车位姿
        
        Args:
            scene_token: 场景token
            
        Returns:
            位姿列表
        """
        sample_tokens = self.get_sample_tokens(scene_token)
        return [self.extract_ego_pose(token) for token in sample_tokens]
    
    def extract_ego_state(self, sample_token: str) -> Dict[str, Any]:
        """
        提取单帧的自车状态（速度、加速度等）
        
        Args:
            sample_token: sample token
            
        Returns:
            自车状态字典，包含速度、加速度、转向角等
            
        原理：
            1. 从CAN bus数据获取车辆监控信息
            2. 匹配最接近的时间戳
            3. 提取速度、转向角、yaw_rate等
        """
        sample = self.nusc.get('sample', sample_token)
        scene_token = sample['scene_token']
        timestamp = sample['timestamp']
        
        # 获取场景名称
        scene = self.nusc.get('scene', scene_token)
        scene_name = scene['name']
        
        # 获取CAN bus数据
        vehicle_messages = self.nusc_can.get_messages(scene_name, 'vehicle_monitor')
        
        # 找到最接近的消息
        can_msg = self._find_closest_message(vehicle_messages, timestamp)
        
        if not can_msg:
            # 如果没有CAN数据，返回默认值
            return {
                'sample_token': sample_token,
                'timestamp': timestamp,
                'velocity': 0.0,  # m/s
                'acceleration': 0.0,  # m/s²
                'steering_angle': 0.0,  # rad
                'yaw_rate': 0.0  # rad/s
            }
        
        # 提取状态信息
        velocity = can_msg['vehicle_speed'] / 3.6  # 从km/h转换为m/s
        steering_angle = np.radians(can_msg['steering'])
        yaw_rate = np.radians(can_msg['yaw_rate'])
        
        # 尝试获取加速度
        acceleration = 0.0
        pose_messages = self.nusc_can.get_messages(scene_name, 'pose')
        pose_msg = self._find_closest_message(pose_messages, timestamp)
        if pose_msg:
            # 纵向加速度（车辆前进方向）
            acceleration = pose_msg['accel'][1]
        
        return {
            'sample_token': sample_token,
            'timestamp': timestamp,
            'velocity': float(velocity),
            'acceleration': float(acceleration),
            'steering_angle': float(steering_angle),
            'yaw_rate': float(yaw_rate)
        }
    
    def extract_ego_states(self, scene_token: str) -> List[Dict[str, Any]]:
        """
        提取场景所有帧的自车状态
        
        Args:
            scene_token: 场景token
            
        Returns:
            自车状态列表
        """
        sample_tokens = self.get_sample_tokens(scene_token)
        return [self.extract_ego_state(token) for token in sample_tokens]
    
    def extract_annotations(
        self,
        sample_token: str,
        coordinate_frame: str = 'global',
        include_ego_relative: bool = True
    ) -> List[Dict[str, Any]]:
        """
        提取单帧的GT标注
        
        Args:
            sample_token: sample token
            coordinate_frame: 坐标系，'global'或'ego'
            include_ego_relative: 是否包含相对于自车的信息（距离、相对速度、可见性）
            
        Returns:
            标注列表，每个标注包含category, translation, size, yaw, velocity等
            
        原理：
            1. 获取sample的所有annotation
            2. 提取每个annotation的属性
            3. 映射类别名称
            4. 计算速度
            5. 根据需要转换坐标系
            6. 计算相对于自车的信息（距离、相对速度、可见性）
        """
        sample = self.nusc.get('sample', sample_token)
        ann_tokens = sample['anns']
        
        # 如果需要ego相对信息，预先获取ego_pose和ego_state
        ego_translation = None
        ego_velocity_vec = None
        if include_ego_relative:
            ego_pose = self.extract_ego_pose(sample_token)
            ego_translation = np.array(ego_pose['translation'])
            ego_state = self.extract_ego_state(sample_token)
            ego_velocity = ego_state['velocity']
            ego_yaw = ego_pose['yaw']
            # 将自车标量速度转换为向量形式（全局坐标系）
            ego_velocity_vec = np.array([
                ego_velocity * np.cos(ego_yaw),
                ego_velocity * np.sin(ego_yaw)
            ])
        
        annotations = []
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # 映射类别
            original_category = ann['category_name']
            if original_category not in CATEGORY_MAPPING:
                continue
            category = CATEGORY_MAPPING[original_category]
            
            # 获取实例ID
            instance_id = self.nusc.getind('instance', ann['instance_token'])
            
            # 位置和旋转（全局坐标系）
            translation = ann['translation']
            rotation = ann['rotation']
            
            # 计算yaw角度
            yaw = quaternion_yaw(Quaternion(rotation))
            
            # 尺寸 [width, length, height]
            size = ann['size']
            
            # 计算速度（全局坐标系）
            velocity = self.nusc.box_velocity(ann_token)
            if np.isnan(velocity[0]):
                velocity = np.array([0.0, 0.0, 0.0])
            
            # 获取属性
            attribute_name = ""
            if ann['attribute_tokens']:
                attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])
                attribute_name = attribute['name']
            
            annotation_dict = {
                'token': ann_token,
                'instance_token': ann['instance_token'],
                'instance_id': instance_id,
                'category': category,
                'translation': translation,
                'rotation': rotation,
                'yaw': float(yaw),
                'size': size,
                'velocity': velocity[:2].tolist(),
                'attribute_name': attribute_name,
                'num_lidar_pts': ann.get('num_lidar_pts', 0),
                'num_radar_pts': ann.get('num_radar_pts', 0)
            }
            
            # 添加相对于自车的信息
            if include_ego_relative:
                # 获取可见性信息
                visibility_token = ann.get('visibility_token', '')
                if visibility_token:
                    visibility = self.nusc.get('visibility', visibility_token)
                    visibility_desc = visibility.get('description', 'unknown')
                    
                    # 尝试获取level字段，如果不存在或不是整数，从description解析
                    visibility_level_raw = visibility.get('level', None)
                    if visibility_level_raw is not None:
                        # 检查是否是整数
                        if isinstance(visibility_level_raw, int):
                            visibility_level = visibility_level_raw
                        elif isinstance(visibility_level_raw, str):
                            # 如果level是字符串，尝试从description解析
                            visibility_level = self._parse_visibility_level(visibility_desc)
                        else:
                            visibility_level = self._parse_visibility_level(visibility_desc)
                    else:
                        # 没有level字段，从description解析
                        visibility_level = self._parse_visibility_level(visibility_desc)
                else:
                    visibility_level = 0
                    visibility_desc = 'unknown'
                
                # 计算与自车的距离（2D欧几里得距离）
                obj_translation = np.array(translation)
                distance = float(np.linalg.norm(obj_translation[:2] - ego_translation[:2]))
                
                # 计算相对速度（全局坐标系）
                obj_velocity = np.array(velocity[:2])
                relative_velocity = obj_velocity - ego_velocity_vec
                
                # 添加到annotation_dict
                annotation_dict.update({
                    'distance_to_ego': distance,
                    'relative_velocity': relative_velocity.tolist(),
                    'visibility_level': visibility_level,
                    'visibility_description': visibility_desc
                })
            
            annotations.append(annotation_dict)
        
        return annotations
    
    def extract_gt_trajectories(
        self,
        sample_token: str,
        future_steps: int = None
    ) -> Dict[str, Any]:
        """
        提取GT轨迹（当前帧所有对象的未来轨迹）
        
        Args:
            sample_token: 当前sample token
            future_steps: 未来时间步数，None表示使用配置的默认值
            
        Returns:
            包含trajectories, masks, labels, instance_ids的字典
            
        原理：
            1. 获取当前帧的所有对象
            2. 对每个对象，追踪其在未来帧的位置
            3. 使用instance_id匹配同一对象
            4. 记录有效性mask（对象可能在某些帧消失）
        """
        if future_steps is None:
            future_steps = config.FUTURE_TRAJECTORY_STEPS
        
        current_annotations = self.extract_annotations(sample_token)
        
        if not current_annotations:
            return {
                'trajectories': np.empty((0, future_steps, 2)),
                'masks': np.empty((0, future_steps)),
                'labels': [],
                'instance_ids': []
            }
        
        trajectories = []
        masks = []
        labels = []
        instance_ids = []
        
        # 为每个对象构建轨迹
        for ann in current_annotations:
            instance_id = ann['instance_id']
            category = ann['category']
            
            trajectory = []
            mask = []
            
            # 当前位置
            current_position = ann['translation'][:2]
            trajectory.append(current_position)
            mask.append(1.0)
            
            # 获取未来位置
            current_sample_token = sample_token
            for step in range(1, future_steps):
                # 获取下一帧
                if current_sample_token:
                    sample = self.nusc.get('sample', current_sample_token)
                    current_sample_token = sample.get('next')
                    
                    if current_sample_token:
                        future_annotations = self.extract_annotations(current_sample_token)
                        
                        # 在未来帧中寻找同一实例
                        found = False
                        for future_ann in future_annotations:
                            if future_ann['instance_id'] == instance_id:
                                trajectory.append(future_ann['translation'][:2])
                                mask.append(1.0)
                                found = True
                                break
                        
                        if not found:
                            # 对象消失，使用最后已知位置
                            trajectory.append(trajectory[-1])
                            mask.append(0.0)
                    else:
                        # 没有更多帧
                        trajectory.append(trajectory[-1])
                        mask.append(0.0)
                else:
                    trajectory.append(trajectory[-1])
                    mask.append(0.0)
            
            # 确保轨迹长度正确
            while len(trajectory) < future_steps:
                trajectory.append(trajectory[-1])
                mask.append(0.0)
            
            trajectory = trajectory[:future_steps]
            mask = mask[:future_steps]
            
            trajectories.append(trajectory)
            masks.append(mask)
            labels.append(category)
            instance_ids.append(instance_id)
        
        return {
            'trajectories': np.array(trajectories) if trajectories else np.empty((0, future_steps, 2)),
            'masks': np.array(masks) if masks else np.empty((0, future_steps)),
            'labels': labels,
            'instance_ids': np.array(instance_ids) if instance_ids else np.empty((0,))
        }
    
    def _parse_visibility_level(self, description: str) -> int:
        """
        从可见性描述字符串中解析出level
        
        Args:
            description: 可见性描述，如'v80-100', 'v60-80', 'v40-60', 'v0-40'
            
        Returns:
            可见性等级 (1-4)，如果无法解析返回0
        """
        if not description or description == 'unknown':
            return 0
        
        # 从description中提取数字范围
        # v80-100 -> level 1
        # v60-80 -> level 2
        # v40-60 -> level 3
        # v0-40 -> level 4
        description_lower = description.lower()
        if 'v80-100' in description_lower or '80-100' in description_lower:
            return 1
        elif 'v60-80' in description_lower or '60-80' in description_lower:
            return 2
        elif 'v40-60' in description_lower or '40-60' in description_lower:
            return 3
        elif 'v0-40' in description_lower or '0-40' in description_lower:
            return 4
        else:
            # 无法解析，返回0
            return 0
    
    def _find_closest_message(
        self,
        messages: List[Dict],
        target_timestamp: int
    ) -> Optional[Dict]:
        """
        从消息列表中找到最接近目标时间戳的消息
        
        Args:
            messages: 消息列表
            target_timestamp: 目标时间戳（微秒）
            
        Returns:
            最接近的消息，如果列表为空返回None
        """
        if not messages:
            return None
        
        closest_msg = min(messages, key=lambda x: abs(x['utime'] - target_timestamp))
        return closest_msg

