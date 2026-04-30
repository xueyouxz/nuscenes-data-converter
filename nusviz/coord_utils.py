"""
坐标转换工具

约定：
- 单位：米（meter）
- 四元数顺序：wxyz（与 nuScenes / pyquaternion 一致）

nuScenes 旋转方向说明：
- ego_pose.rotation：描述 ego → world 的旋转变换
  （从 ego 坐标系旋转到世界坐标系）
- calibrated_sensor.rotation：描述 sensor → ego 的旋转变换
  （从传感器坐标系旋转到 ego 坐标系）

投影方向（世界 → 图像）需要取逆：
- T_world_ego = (T_ego_world)⁻¹
- T_ego_cam  = (T_cam_ego)⁻¹
"""

import numpy as np
from pyquaternion import Quaternion
from typing import List


def quat_to_wxyz(q: Quaternion) -> List[float]:
    """将 pyquaternion.Quaternion 转换为 [w, x, y, z] 列表。"""
    return [q.w, q.x, q.y, q.z]


def transform_points_to_world(
    points: np.ndarray,
    ego_translation: List[float],
    ego_rotation: Quaternion,
    sensor_translation: List[float],
    sensor_rotation: Quaternion,
) -> np.ndarray:
    """
    将点云从传感器坐标系变换到世界坐标系。

    变换链：sensor -> ego -> world

    Args:
        points: (N, 3) 传感器坐标系下的点
        ego_translation: ego_pose.translation
        ego_rotation: ego_pose.rotation
        sensor_translation: calibrated_sensor.translation
        sensor_rotation: calibrated_sensor.rotation

    Returns:
        (N, 3) 世界坐标系下的点
    """
    # sensor -> ego
    points_ego = (sensor_rotation.rotation_matrix @ points.T).T + np.array(sensor_translation)
    # ego -> world
    points_world = (ego_rotation.rotation_matrix @ points_ego.T).T + np.array(ego_translation)
    return points_world

