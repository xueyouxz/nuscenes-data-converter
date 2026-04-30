"""
坐标转换工具模块

提供自车坐标系与全局坐标系之间的转换功能
基于temp/utils/utils.py重构并优化
"""

import numpy as np
from typing import Union, List
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction import convert_global_coords_to_local
from nuscenes.prediction.helper import convert_local_coords_to_global


def ensure_quaternion(rotation_data: Union[List, Quaternion]) -> Quaternion:
    """
    确保旋转数据是Quaternion对象
    
    Args:
        rotation_data: 旋转数据，可以是列表或Quaternion对象
        
    Returns:
        Quaternion对象
        
    原理：
        统一处理不同格式的旋转数据，确保后续计算使用正确的Quaternion对象
    """
    if isinstance(rotation_data, Quaternion):
        return rotation_data
    elif isinstance(rotation_data, (list, tuple)):
        return Quaternion(rotation_data)
    elif hasattr(rotation_data, 'tolist'):
        # 处理numpy数组
        return Quaternion(rotation_data.tolist())
    else:
        # 默认返回单位四元数
        return Quaternion([1.0, 0.0, 0.0, 0.0])


def transform_to_global(
    local_coords: np.ndarray,
    ego_translation: Union[List, np.ndarray],
    ego_rotation: Union[List, Quaternion]
) -> np.ndarray:
    """
    将局部坐标（Ego坐标系）转换到全局坐标系
    
    Args:
        local_coords: 局部坐标，shape: (N, 2) 或 (N, 3)
        ego_translation: 自车全局位置 [x, y, z]
        ego_rotation: 自车全局旋转四元数 [w, x, y, z]
        
    Returns:
        全局坐标，shape与输入相同
        
    原理：
        1. 坐标变换公式：global = R * local + T
           - R是旋转矩阵（由四元数转换得到）
           - T是平移向量（自车位置）
        2. 只对x, y坐标进行变换，z坐标保持不变
        3. 使用nuscenes官方API确保计算正确性
    """
    local_coords = np.array(local_coords)
    ego_rotation = ensure_quaternion(ego_rotation)
    
    # 处理单个点的情况
    if local_coords.ndim == 1:
        local_coords = local_coords.reshape(1, -1)
    
    # 只取x, y坐标进行转换
    coords_2d = local_coords[:, :2]
    
    # 使用nuscenes API转换到全局坐标系
    global_coords_2d = convert_local_coords_to_global(
        coords_2d, ego_translation, ego_rotation
    )
    
    # 如果原始数据有z坐标，保持不变
    if local_coords.shape[1] == 3:
        z_coords = local_coords[:, 2:3]
        return np.concatenate([global_coords_2d, z_coords], axis=1)
    else:
        return global_coords_2d


def transform_to_local(
    global_coords: np.ndarray,
    ego_translation: Union[List, np.ndarray],
    ego_rotation: Union[List, Quaternion]
) -> np.ndarray:
    """
    将全局坐标转换到局部坐标系（Ego坐标系）
    
    Args:
        global_coords: 全局坐标，shape: (N, 2) 或 (N, 3)
        ego_translation: 自车全局位置 [x, y, z]
        ego_rotation: 自车全局旋转四元数 [w, x, y, z]
        
    Returns:
        局部坐标，shape与输入相同
        
    原理：
        1. 逆变换公式：local = R^(-1) * (global - T)
        2. 只对x, y坐标进行变换
        3. 使用nuscenes官方API确保正确性
    """
    global_coords = np.array(global_coords)
    ego_rotation = ensure_quaternion(ego_rotation)
    
    # 处理单个点的情况
    if global_coords.ndim == 1:
        global_coords = global_coords.reshape(1, -1)
    
    # 只取x, y坐标进行转换
    coords_2d = global_coords[:, :2]
    
    # 使用nuscenes API转换到局部坐标系
    local_coords_2d = convert_global_coords_to_local(
        coords_2d, ego_translation, ego_rotation
    )
    
    # 如果原始数据有z坐标，保持不变
    if global_coords.shape[1] == 3:
        z_coords = global_coords[:, 2:3]
        return np.concatenate([local_coords_2d, z_coords], axis=1)
    else:
        return local_coords_2d


def batch_transform_coords(
    coords: np.ndarray,
    ego_translation: Union[List, np.ndarray],
    ego_rotation: Union[List, Quaternion],
    to_global: bool = True
) -> np.ndarray:
    """
    批量转换坐标
    
    Args:
        coords: 坐标数组，shape: (N, 2) 或 (N, 3)
        ego_translation: 自车全局位置
        ego_rotation: 自车全局旋转
        to_global: True表示局部到全局，False表示全局到局部
        
    Returns:
        转换后的坐标
        
    原理：
        统一接口，根据to_global参数选择转换方向
        使用向量化操作提高批量处理效率
    """
    if to_global:
        return transform_to_global(coords, ego_translation, ego_rotation)
    else:
        return transform_to_local(coords, ego_translation, ego_rotation)


def transform_yaw_to_global(
    local_yaw: Union[float, np.ndarray],
    ego_rotation: Union[List, Quaternion]
) -> Union[float, np.ndarray]:
    """
    将局部yaw角度转换到全局坐标系
    
    Args:
        local_yaw: 局部yaw角度（弧度），标量或数组
        ego_rotation: 自车全局旋转四元数
        
    Returns:
        全局yaw角度（弧度）
        
    原理：
        yaw角度的转换：global_yaw = ego_yaw + local_yaw
        - ego_yaw是自车在全局坐标系的朝向
        - 通过四元数计算得到
    """
    ego_rotation = ensure_quaternion(ego_rotation)
    ego_yaw = quaternion_yaw(ego_rotation)
    
    if isinstance(local_yaw, np.ndarray):
        return local_yaw + ego_yaw
    else:
        return float(local_yaw + ego_yaw)


def transform_yaw_to_local(
    global_yaw: Union[float, np.ndarray],
    ego_rotation: Union[List, Quaternion]
) -> Union[float, np.ndarray]:
    """
    将全局yaw角度转换到局部坐标系
    
    Args:
        global_yaw: 全局yaw角度（弧度），标量或数组
        ego_rotation: 自车全局旋转四元数
        
    Returns:
        局部yaw角度（弧度）
        
    原理：
        逆变换：local_yaw = global_yaw - ego_yaw
    """
    ego_rotation = ensure_quaternion(ego_rotation)
    ego_yaw = quaternion_yaw(ego_rotation)
    
    if isinstance(global_yaw, np.ndarray):
        return global_yaw - ego_yaw
    else:
        return float(global_yaw - ego_yaw)


def batch_transform_yaw(
    yaws: np.ndarray,
    ego_rotation: Union[List, Quaternion],
    to_global: bool = True
) -> np.ndarray:
    """
    批量转换yaw角度
    
    Args:
        yaws: yaw角度数组（弧度）
        ego_rotation: 自车全局旋转
        to_global: True表示局部到全局，False表示全局到局部
        
    Returns:
        转换后的yaw角度数组
        
    原理：
        向量化处理多个yaw角度，提高批量转换效率
    """
    if to_global:
        return transform_yaw_to_global(yaws, ego_rotation)
    else:
        return transform_yaw_to_local(yaws, ego_rotation)


def transform_velocity_to_global(
    local_velocity: np.ndarray,
    ego_rotation: Union[List, Quaternion]
) -> np.ndarray:
    """
    将局部速度向量转换到全局坐标系
    
    Args:
        local_velocity: 局部速度向量 [vx, vy, vz]
        ego_rotation: 自车全局旋转四元数
        
    Returns:
        全局速度向量
        
    原理：
        速度向量的变换只需要旋转，不需要平移
        使用旋转矩阵：global_v = R * local_v
    """
    ego_rotation = ensure_quaternion(ego_rotation)
    rotation_matrix = ego_rotation.rotation_matrix
    
    local_velocity = np.array(local_velocity)
    global_velocity = rotation_matrix.dot(local_velocity)
    
    return global_velocity


def transform_velocity_to_local(
    global_velocity: np.ndarray,
    ego_rotation: Union[List, Quaternion]
) -> np.ndarray:
    """
    将全局速度向量转换到局部坐标系
    
    Args:
        global_velocity: 全局速度向量 [vx, vy, vz]
        ego_rotation: 自车全局旋转四元数
        
    Returns:
        局部速度向量
        
    原理：
        使用逆旋转矩阵：local_v = R^(-1) * global_v
    """
    ego_rotation = ensure_quaternion(ego_rotation)
    ego_rotation_inv = ego_rotation.inverse
    rotation_matrix_inv = ego_rotation_inv.rotation_matrix
    
    global_velocity = np.array(global_velocity)
    local_velocity = rotation_matrix_inv.dot(global_velocity)
    
    return local_velocity

