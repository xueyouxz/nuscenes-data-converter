"""
相机流构建器

生成camera_stream.json文件并复制相机图片
包含所有帧的相机数据、内参、外参和图片路径
"""

import shutil
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
from pyquaternion import Quaternion

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline
from ..config import config


class CameraStreamBuilder(BaseBuilder):
    """
    相机流构建器
    
    构建场景的相机流文件（camera_stream.json）并组织相机图片
    """
    
    def __init__(self, nuscenes_extractor=None):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器
        """
        self.nusc_extractor = nuscenes_extractor
        self.camera_channels = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
    
    def build(
        self,
        scene_token_or_pipeline: Union[str, ScenePipeline],
        output_dir: Path = None,
        scene_name: str = None
    ) -> Dict[str, Any]:
        """
        构建相机流数据并复制图片
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            output_dir: 输出目录（必需）
            scene_name: 场景名称（必需）
            
        Returns:
            相机流数据字典
        
        注意：虽然output_dir和scene_name是可选参数，但实际使用时必须提供
        """
        if output_dir is None or scene_name is None:
            raise ValueError("output_dir和scene_name参数是必需的")
        # 获取sample tokens
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            sample_tokens = scene_token_or_pipeline.get_sample_tokens()
            scene_token = scene_token_or_pipeline.scene_token
            nusc_extractor = scene_token_or_pipeline.nusc_extractor
        else:
            scene_token = scene_token_or_pipeline
            sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
            nusc_extractor = self.nusc_extractor
        
        # 创建图片目录（按相机通道组织）
        scene_dir = output_dir / scene_name
        images_dir = scene_dir / 'images'
        
        if config.COPY_CAMERA_IMAGES:
            images_dir.mkdir(parents=True, exist_ok=True)
            for channel in self.camera_channels:
                channel_dir = images_dir / channel
                channel_dir.mkdir(exist_ok=True)
        
        # 构建所有帧的数据
        frames = []
        for frame_index, sample_token in enumerate(sample_tokens):
            frame_data = self._build_frame(
                sample_token,
                frame_index,
                images_dir,
                nusc_extractor
            )
            frames.append(frame_data)
        
        return {
            'scene_token': scene_token,
            'frames': frames
        }
    
    def _build_frame(
        self,
        sample_token: str,
        frame_index: int,
        images_dir: Path,
        nusc_extractor
    ) -> Dict[str, Any]:
        """
        构建单帧相机数据
        
        Args:
            sample_token: sample token
            frame_index: 帧索引
            images_dir: 图片目录
            nusc_extractor: NuScenes提取器
            
        Returns:
            帧数据字典
        """
        extractor = nusc_extractor or self.nusc_extractor
        sample = extractor.nusc.get('sample', sample_token)
        ego_pose = extractor.extract_ego_pose(sample_token)
        
        cameras = []
        for cam_channel in self.camera_channels:
            cam_token = sample['data'][cam_channel]
            cam_data = self._extract_camera_data(
                cam_token,
                cam_channel,
                frame_index,
                images_dir,
                extractor
            )
            cameras.append(cam_data)
        
        return {
            'frame_index': frame_index,
            'timestamp': ego_pose['timestamp'],
            'cameras': cameras
        }
    
    def _extract_camera_data(
        self,
        cam_token: str,
        cam_channel: str,
        frame_index: int,
        images_dir: Path,
        nusc_extractor
    ) -> Dict[str, Any]:
        """
        提取单个相机的数据并复制图片
        
        Args:
            cam_token: 相机token
            cam_channel: 相机通道
            frame_index: 帧索引
            images_dir: 图片目录
            nusc_extractor: NuScenes提取器
            
        Returns:
            相机数据字典
        """
        extractor = nusc_extractor or self.nusc_extractor
        sd_rec = extractor.nusc.get('sample_data', cam_token)
        cs_rec = extractor.nusc.get(
            'calibrated_sensor',
            sd_rec['calibrated_sensor_token']
        )
        
        # 图片路径处理
        relative_path = f'images/{cam_channel}/frame_{frame_index:03d}.jpg'
        
        if config.COPY_CAMERA_IMAGES:
            src_image_path = Path(extractor.dataroot) / sd_rec['filename']
            dst_image_path = images_dir / cam_channel / f'frame_{frame_index:03d}.jpg'
            
            if src_image_path.exists():
                if config.COMPRESS_CAMERA_IMAGES:
                    self._compress_and_save_image(src_image_path, dst_image_path)
                else:
                    shutil.copy2(src_image_path, dst_image_path)
        
        # 提取相机参数
        intrinsic = np.array(cs_rec['camera_intrinsic'])
        translation = np.array(cs_rec['translation'])
        rotation = Quaternion(cs_rec['rotation'])
        
        # 预计算变换矩阵
        ego_to_camera = self._compute_ego_to_camera_matrix(translation, rotation)
        
        return {
            'channel': cam_channel,
            'image_path': relative_path,
            'width': sd_rec['width'],
            'height': sd_rec['height'],
            'intrinsic': intrinsic.tolist(),
            'extrinsic': {
                'translation': translation.tolist(),
                'rotation': [rotation.w, rotation.x, rotation.y, rotation.z]
            },
            'ego_to_camera': ego_to_camera.tolist()
        }
    
    def _compute_ego_to_camera_matrix(
        self,
        translation: np.ndarray,
        rotation: Quaternion
    ) -> np.ndarray:
        """
        计算从ego坐标系到camera坐标系的变换矩阵
        
        Args:
            translation: 相机在ego中的平移
            rotation: 相机在ego中的旋转
            
        Returns:
            4x4变换矩阵
        """
        # camera在ego中的位姿
        R = rotation.rotation_matrix
        t = translation
        
        # ego到camera的变换（求逆）
        ego_to_cam = np.eye(4)
        ego_to_cam[:3, :3] = R.T
        ego_to_cam[:3, 3] = -R.T @ t
        
        return ego_to_cam
    
    def _compress_and_save_image(
        self,
        src_path: Path,
        dst_path: Path
    ):
        """
        压缩并保存图片
        
        Args:
            src_path: 源图片路径
            dst_path: 目标图片路径
        """
        from PIL import Image
        
        img = Image.open(src_path)
        img.save(
            dst_path,
            'JPEG',
            quality=config.CAMERA_IMAGE_QUALITY,
            optimize=True
        )
    
    def save(
        self,
        camera_stream: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存相机流到JSON文件
        
        Args:
            camera_stream: 相机流数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_json(camera_stream, output_dir, scene_name, 'camera_stream.json')

