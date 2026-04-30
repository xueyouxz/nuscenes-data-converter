"""
Message 构建器

生成 messages/*.glb 文件（nuviz/state_update 消息）。
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import io

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from glb_encoder import GLBEncoder
from coord_utils import quat_to_wxyz, transform_points_to_world


CAMERA_CHANNELS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# nuScenes 层级类别名 -> 简化名映射
_CATEGORY_MAP = {
    'vehicle.car':          'car',
    'vehicle.truck':        'truck',
    'vehicle.bus':          'bus',
    'vehicle.trailer':      'trailer',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle':   'motorcycle',
    'vehicle.bicycle':      'bicycle',
    'pedestrian':           'pedestrian',
    'barrier':              'barrier',
    'traffic_cone':         'traffic_cone',
}

# 简化类别名 -> ID 映射（仅检测类）
_CATEGORY_TO_ID = {
    "barrier":              1,
    "bicycle":              2,
    "bus":                  3,
    "car":                  4,
    "construction_vehicle": 5,
    "motorcycle":           6,
    "pedestrian":           7,
    "traffic_cone":         8,
    "trailer":              9,
    "truck":                10,
}


def _resolve_category(category_name: str) -> str:
    """将 nuScenes 完整类别名解析为简化名。"""
    for prefix, simple_name in _CATEGORY_MAP.items():
        if prefix in category_name:
            return simple_name
    return category_name.split('.')[0]


class MessageBuilder:
    """将单个 NuScenes sample 编码为 nuviz/state_update GLB 文件。"""

    def __init__(self, nusc: NuScenes, dataroot: str):
        self.nusc = nusc
        self.dataroot = Path(dataroot)

    def build_message(
        self,
        sample_token: str,
        update_type: str = "INCREMENTAL",
        include_lidar: bool = True,
        include_cameras: bool = True,
        include_objects: bool = True,
        ego_future_poses: Optional[List[List[float]]] = None,
        obj_future_trajectories: Optional[List[List[List[float]]]] = None,
    ) -> bytes:
        """
        构建单条 state_update 消息并返回 GLB 字节。

        Args:
            sample_token: NuScenes sample token
            update_type:  "COMPLETE_STATE" 或 "INCREMENTAL"
            include_lidar:   是否包含点云
            include_cameras: 是否包含相机图像
            include_objects: 是否包含目标框
            ego_future_poses: 自车未来轨迹点列表，每项 [x,y,z]，从当前帧（含）到末帧
            obj_future_trajectories: 对象未来轨迹列表，顺序与 sample['anns'] 对应，
                                     每项为该对象的未来中心点列表 [[x,y,z], ...]
        """
        sample = self.nusc.get('sample', sample_token)
        encoder = GLBEncoder()

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = self.nusc.get('sample_data', lidar_token)
        ego_pose    = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        ego_translation = ego_pose['translation']
        ego_rotation    = Quaternion(ego_pose['rotation'])

        update: Dict[str, Any] = {
            "timestamp": sample['timestamp'] / 1e6,
            "poses": {
                "/ego_pose": {
                    "translation": ego_translation,
                    "rotation":    quat_to_wxyz(ego_rotation),
                }
            },
            "primitives": {},
        }

        if include_lidar:
            prim = self._build_lidar(encoder, lidar_token, ego_translation, ego_rotation)
            if prim:
                update["primitives"]["/lidar"] = prim

        if include_objects:
            prim = self._build_objects(encoder, sample)
            if prim:
                update["primitives"]["/objects/bounds"] = prim

        if include_cameras:
            for channel in CAMERA_CHANNELS:
                if channel not in sample['data']:
                    continue
                prim = self._build_camera(encoder, sample['data'][channel])
                if prim:
                    update["primitives"][f"/camera/{channel}"] = prim

        if ego_future_poses is not None:
            prim = self._build_ego_fut_trajectory(encoder, ego_future_poses)
            if prim:
                update["primitives"]["/ego/fut_trajectory"] = prim

        if obj_future_trajectories is not None:
            prim = self._build_objects_fut_trajectories(encoder, obj_future_trajectories)
            if prim:
                update["primitives"]["/objects/fut_trajectories"] = prim

        state_update = {
            "type": "nuviz/state_update",
            "data": {
                "update_type": update_type,
                "updates":     [update],
            },
        }
        return encoder.encode(state_update)

    def _build_lidar(
        self,
        encoder: GLBEncoder,
        lidar_token: str,
        ego_translation: List[float],
        ego_rotation: Quaternion,
    ) -> Optional[Dict[str, Any]]:
        """读取 LiDAR 点云并转换到世界坐标系，编码为 GLB accessor。"""
        lidar_data = self.nusc.get('sample_data', lidar_token)
        cs_rec     = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        points_raw = np.fromfile(str(self.dataroot / lidar_data['filename']), dtype=np.float32).reshape(-1, 5)
        points_xyz = points_raw[:, :3]
        intensity  = points_raw[:, 3]

        points_world = transform_points_to_world(
            points_xyz,
            ego_translation, ego_rotation,
            cs_rec['translation'], Quaternion(cs_rec['rotation']),
        )

        points_acc    = encoder.add_accessor(points_world.astype(np.float32), type_str="VEC3")
        intensity_acc = encoder.add_accessor(intensity.astype(np.float32),   type_str="SCALAR")

        return {
            "points": [{
                "points": f"#/accessors/{points_acc}",
                "extensions": {
                    "nuscenes": {"INTENSITY": f"#/accessors/{intensity_acc}"}
                },
            }]
        }

    def _build_objects(
        self,
        encoder: GLBEncoder,
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """将 sample 的标注框编码为 cuboids accessor（标注已在世界坐标系下）。"""
        anns = sample['anns']
        if not anns:
            return None

        centers, sizes, rotations, class_ids, track_ids = [], [], [], [], []

        for ann_token in anns:
            ann = self.nusc.get('sample_annotation', ann_token)

            centers.append(np.array(ann['translation']))
            sizes.append(np.array(ann['size']))
            rotations.append(quat_to_wxyz(Quaternion(ann['rotation'])))

            category_name = _resolve_category(ann['category_name'])
            class_ids.append(_CATEGORY_TO_ID.get(category_name, 0))
            track_ids.append(self.nusc.getind('instance', ann['instance_token']))

        center_acc   = encoder.add_accessor(np.array(centers,   dtype=np.float32), type_str="VEC3")
        size_acc     = encoder.add_accessor(np.array(sizes,     dtype=np.float32), type_str="VEC3")
        rotation_acc = encoder.add_accessor(np.array(rotations, dtype=np.float32), type_str="VEC4")
        class_id_acc = encoder.add_accessor(np.array(class_ids, dtype=np.uint32),  type_str="SCALAR")
        track_id_acc = encoder.add_accessor(np.array(track_ids, dtype=np.uint32),  type_str="SCALAR")

        return {
            "cuboids": [{
                "count":    len(centers),
                "CENTER":   f"#/accessors/{center_acc}",
                "SIZE":     f"#/accessors/{size_acc}",
                "ROTATION": f"#/accessors/{rotation_acc}",
                "CLASS_ID": f"#/accessors/{class_id_acc}",
                "TRACK_ID": f"#/accessors/{track_id_acc}",
            }]
        }

    def _build_camera(
        self,
        encoder: GLBEncoder,
        cam_token: str,
    ) -> Optional[Dict[str, Any]]:
        """读取相机图像，压缩为 JPEG 并写入 GLB。"""
        cam_data = self.nusc.get('sample_data', cam_token)
        img      = Image.open(self.dataroot / cam_data['filename'])
        width, height = img.size

        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=85)

        image_idx = encoder.add_image(buf.getvalue(), "image/jpeg", width, height)
        return {"images": [{"data": f"#/images/{image_idx}"}]}

    def _build_ego_fut_trajectory(
        self,
        encoder: GLBEncoder,
        ego_future_poses: List[List[float]],
    ) -> Optional[Dict[str, Any]]:
        """
        将自车未来轨迹编码为 VEC3 accessor。

        Args:
            encoder: GLBEncoder 实例
            ego_future_poses: 从当前帧（含）到末帧的自车世界坐标位置列表，
                              每项 [x, y, z]，顺序与 message_index 时间顺序一致。
        Returns:
            /ego/fut_trajectory primitive JSON，或 None（列表为空时）。
        """
        if not ego_future_poses:
            return None

        poses_arr = np.array(ego_future_poses, dtype=np.float32)  # (M, 3)
        poses_acc = encoder.add_accessor(poses_arr, type_str="VEC3")

        return {
            "trajectory": [{
                "poses": f"#/accessors/{poses_acc}",
                "count": len(ego_future_poses),
            }]
        }

    def _build_objects_fut_trajectories(
        self,
        encoder: GLBEncoder,
        obj_future_trajectories: List[List[List[float]]],
    ) -> Optional[Dict[str, Any]]:
        """
        将所有当前帧对象的未来轨迹编码为 CSR 格式 accessor。

        points  (T, 3) float32 — 所有对象轨迹点拼接
        offsets (M+1,) uint32  — CSR 偏移，第 i 对象轨迹点为
                                 points[offsets[i]:offsets[i+1]]

        Args:
            encoder: GLBEncoder 实例
            obj_future_trajectories: 长度 M 的列表，与 /objects/bounds cuboids
                                     顺序严格对应；每项为 [[x,y,z], ...]，
                                     无未来出现时为空列表。
        Returns:
            /objects/fut_trajectories primitive JSON，或 None（对象列表为空时）。
        """
        obj_count = len(obj_future_trajectories)
        if obj_count == 0:
            return None

        all_points: List[List[float]] = []
        offsets: List[int] = [0]

        for traj in obj_future_trajectories:
            all_points.extend(traj)
            offsets.append(len(all_points))

        if all_points:
            points_arr = np.array(all_points, dtype=np.float32)  # (T, 3)
        else:
            points_arr = np.zeros((0, 3), dtype=np.float32)

        offsets_arr = np.array(offsets, dtype=np.uint32)          # (M+1,)

        points_acc  = encoder.add_accessor(points_arr,  type_str="VEC3")
        offsets_acc = encoder.add_accessor(offsets_arr, type_str="SCALAR")

        return {
            "trajectories": [{
                "points":    f"#/accessors/{points_acc}",
                "offsets":   f"#/accessors/{offsets_acc}",
                "obj_count": obj_count,
            }]
        }
