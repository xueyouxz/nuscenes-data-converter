"""
预测流数据构建器

生成prediction_stream.bin文件（MessagePack格式）
包含所有帧的模型预测数据
"""

from typing import Dict, Any, Union
from pathlib import Path

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline


class PredictionStreamBuilder(BaseBuilder):
    """
    预测流数据构建器
    
    构建场景的预测流文件（prediction_stream.bin）
    """
    
    def __init__(
        self,
        nuscenes_extractor=None,
        sparsedrive_extractor=None
    ):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器
            sparsedrive_extractor: SparseDrive预测提取器
        """
        self.nusc_extractor = nuscenes_extractor
        self.sd_extractor = sparsedrive_extractor
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建预测流数据
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            预测流数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            pipeline = scene_token_or_pipeline
            sample_tokens = pipeline.get_sample_tokens()
            scene_token = pipeline.scene_token
            nusc_extractor = pipeline.nusc_extractor
            sd_extractor = pipeline.sd_extractor
        else:
            scene_token = scene_token_or_pipeline
            sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
            nusc_extractor = self.nusc_extractor
            sd_extractor = self.sd_extractor
        
        # 构建所有帧的数据
        frames = []
        for frame_index, sample_token in enumerate(sample_tokens):
            # 检查是否有预测
            if not sd_extractor.has_prediction(sample_token):
                continue
            
            frame_data = self._build_frame(
                sample_token, frame_index, nusc_extractor, sd_extractor
            )
            frames.append(frame_data)
        
        pred_stream = {
            'scene_token': scene_token,
            'frames': frames
        }
        
        return pred_stream
    
    def _build_frame(
        self,
        sample_token: str,
        frame_index: int,
        nusc_extractor=None,
        sd_extractor=None
    ) -> Dict[str, Any]:
        """
        构建单帧预测数据
        
        Args:
            sample_token: sample token
            frame_index: 帧索引
            
        Returns:
            帧数据字典
            
        原理：
            提取：
            - detection: 检测框（boxes, scores, classes, track_ids, trajectories）
            - planning: 规划轨迹
            - mapping: 在线地图预测
            所有坐标都在全局坐标系中
        """
        nusc_ext = nusc_extractor or self.nusc_extractor
        sd_ext = sd_extractor or self.sd_extractor
        
        # 提取时间戳
        ego_pose = nusc_ext.extract_ego_pose(sample_token)
        
        # 提取检测
        detections = sd_ext.extract_detections(sample_token)
        
        # 组织检测数据
        boxes = []
        scores = []
        classes = []
        track_ids = []
        trajectories = []
        
        for det in detections:
            # box: [x, y, z, w, l, h, yaw]
            trans = det['translation']
            size = det['size']
            yaw = det['yaw']
            box = [
                trans[0], trans[1], 0.0,  # z设为0
                size[0], size[1], size[2],
                yaw
            ]
            boxes.append(box)
            scores.append(det['score'])
            classes.append(det['category'])
            track_ids.append(str(det['instance_id']))
            trajectories.append(det['trajectories'])  # 6个模态，每个12步
        
        # 提取规划
        planning = sd_ext.extract_planning(sample_token)
        
        # 提取地图预测
        map_predictions = sd_ext.extract_map_predictions(sample_token)
        
        # 组织地图数据
        dividers = []
        boundaries = []
        ped_crossings = []
        
        for map_pred in map_predictions:
            category = map_pred['category']
            vectors = map_pred['vectors']
            
            if category == 'divider':
                dividers.append(vectors)
            elif category == 'boundary':
                boundaries.append(vectors)
            elif category == 'ped_crossing':
                ped_crossings.append(vectors)
        
        frame_data = {
            'frame_index': frame_index,
            'timestamp': ego_pose['timestamp'],
            'detection': {
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'track_ids': track_ids,
                'trajectories': trajectories
            },
            'planning': planning,
            'mapping': {
                'dividers': dividers,
                'boundaries': boundaries,
                'ped_crossings': ped_crossings
            }
        }
        
        return frame_data
    
    def save(
        self,
        pred_stream: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存预测流到MessagePack文件
        
        Args:
            pred_stream: 预测流数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_msgpack(pred_stream, output_dir, scene_name, 'prediction_stream.bin')

