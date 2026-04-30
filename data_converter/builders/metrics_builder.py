"""
指标构建器（优化版本）

生成metrics.json文件
包含每帧和场景级别的评估指标

优化说明：
- 复用pipeline中缓存的地图数据
- 复用pipeline中缓存的匹配结果
- 消除重复的地图提取计算
"""

from typing import Dict, Any, List, Union
from pathlib import Path
import numpy as np

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline
from ..core.evaluators.detection import DetectionEvaluator
from ..core.evaluators.mapping import MappingEvaluator
from ..core.evaluators.motion import MotionEvaluator
from ..core.evaluators.planning import PlanningEvaluator
from ..utils.statistics import aggregate_metrics


class MetricsBuilder(BaseBuilder):
    """
    指标构建器
    
    构建场景的指标文件（metrics.json）
    """
    
    def __init__(
        self,
        nuscenes_extractor=None,
        sparsedrive_extractor=None,
        map_extractor=None
    ):
        """
        初始化构建器
        
        Args:
            nuscenes_extractor: NuScenes数据提取器（传统模式）
            sparsedrive_extractor: SparseDrive预测提取器（传统模式）
            map_extractor: 地图提取器（传统模式）
        """
        self.nusc_extractor = nuscenes_extractor
        self.sd_extractor = sparsedrive_extractor
        self.map_extractor = map_extractor
        
        # 初始化评估器
        self.detection_evaluator = DetectionEvaluator()
        self.mapping_evaluator = MappingEvaluator()
        self.motion_evaluator = MotionEvaluator()
        self.planning_evaluator = PlanningEvaluator()
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建指标数据
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            指标数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            return self._build_from_pipeline(scene_token_or_pipeline)
        else:
            return self._build_traditional(scene_token_or_pipeline)
    
    def _build_from_pipeline(self, pipeline: ScenePipeline) -> Dict[str, Any]:
        """
        从pipeline构建指标（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            
        Returns:
            指标数据字典
        """
        sample_tokens = pipeline.get_sample_tokens()
        
        # 评估所有帧（复用pipeline的缓存）
        frame_metrics = self._evaluate_all_frames_from_pipeline(pipeline, sample_tokens)
        
        # 计算场景汇总
        scene_summary = self._compute_scene_summary(frame_metrics)
        
        metrics = {
            'scene_token': pipeline.scene_token,
            'scene_summary': scene_summary,
            'frame_metrics': frame_metrics
        }
        
        return metrics
    
    def _build_traditional(self, scene_token: str) -> Dict[str, Any]:
        """
        传统方式构建指标（向后兼容）
        
        Args:
            scene_token: 场景token
            
        Returns:
            指标数据字典
        """
        sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
        
        # 评估所有帧
        frame_metrics = self._evaluate_all_frames_traditional(scene_token, sample_tokens)
        
        # 计算场景汇总
        scene_summary = self._compute_scene_summary(frame_metrics)
        
        metrics = {
            'scene_token': scene_token,
            'scene_summary': scene_summary,
            'frame_metrics': frame_metrics
        }
        
        return metrics
    
    def _evaluate_all_frames_from_pipeline(
        self,
        pipeline: ScenePipeline,
        sample_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        从pipeline评估所有帧（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            sample_tokens: sample token列表
            
        Returns:
            每帧指标列表
        """
        # 预先获取静态地图（复用pipeline缓存）
        static_map = pipeline.get_static_map()
        
        frame_metrics = []
        
        for frame_index, sample_token in enumerate(sample_tokens):
            if not pipeline.sd_extractor.has_prediction(sample_token):
                continue
            
            metrics = self._evaluate_frame_from_pipeline(
                pipeline, frame_index, sample_token, static_map
            )
            frame_metrics.append(metrics)
        
        return frame_metrics
    
    def _evaluate_all_frames_traditional(
        self,
        scene_token: str,
        sample_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        传统方式评估所有帧
        
        Args:
            scene_token: 场景token
            sample_tokens: sample token列表
            
        Returns:
            每帧指标列表
        """
        # 预先提取地图
        ego_poses = self.nusc_extractor.extract_ego_poses(scene_token)
        static_map = self.map_extractor.extract_static_map(scene_token, ego_poses)
        
        frame_metrics = []
        
        for frame_index, sample_token in enumerate(sample_tokens):
            if not self.sd_extractor.has_prediction(sample_token):
                continue
            
            metrics = self._evaluate_frame_traditional(
                sample_token, frame_index, static_map
            )
            frame_metrics.append(metrics)
        
        return frame_metrics
    
    def _evaluate_frame_from_pipeline(
        self,
        pipeline: ScenePipeline,
        frame_index: int,
        sample_token: str,
        static_map: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        从pipeline评估单帧（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            frame_index: 帧索引
            sample_token: sample token
            static_map: 静态地图
            
        Returns:
            帧指标字典
        """
        ego_pose = pipeline.nusc_extractor.extract_ego_pose(sample_token)
        
        # 从pipeline获取缓存的数据
        pred_detections = pipeline.get_pred_detections(frame_index, sample_token)
        gt_annotations = pipeline.get_gt_annotations(frame_index, sample_token)
        pred_map = pipeline.get_pred_map(frame_index, sample_token)
        gt_trajectories = pipeline.get_gt_trajectories(frame_index, sample_token)
        
        # 获取规划数据
        pred_planning = pipeline.sd_extractor.extract_planning(sample_token)
        gt_planning_list = self._extract_gt_planning(
            sample_token, pred_planning, pipeline.nusc_extractor
        )
        
        # 评估检测
        detection_metrics = self.detection_evaluator.evaluate_sample(
            pred_detections, gt_annotations
        )
        
        # 移除详细对象指标，添加统计信息
        per_object_metrics = detection_metrics.pop('per_object_metrics', [])
        detection_metrics['num_tp'] = len([m for m in per_object_metrics if m.get('is_tp', False)])
        detection_metrics['num_fp'] = len([m for m in per_object_metrics if not m.get('is_tp', False)])
        num_gt = len(gt_annotations)
        detection_metrics['num_fn'] = max(0, num_gt - detection_metrics['num_tp'])
        
        # 评估地图
        mapping_metrics = self.mapping_evaluator.evaluate_sample(
            pred_map, static_map
        )
        mapping_metrics.pop('prediction_errors', None)
        
        # 评估运动
        motion_metrics = self.motion_evaluator.evaluate_sample(
            pred_detections, gt_trajectories
        )
        
        # 评估规划
        planning_metrics = self.planning_evaluator.evaluate_sample(
            pred_planning, gt_planning_list, static_map, gt_annotations
        )
        
        # 组合指标
        frame_metrics = {
            'frame_index': frame_index,
            'timestamp': ego_pose['timestamp'],
            'detection': detection_metrics,
            'mapping': mapping_metrics,
            'motion': motion_metrics,
            'planning': planning_metrics
        }
        
        return frame_metrics
    
    def _evaluate_frame_traditional(
        self,
        sample_token: str,
        frame_index: int,
        static_map: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        传统方式评估单帧
        
        Args:
            sample_token: sample token
            frame_index: 帧索引
            static_map: 静态地图
            
        Returns:
            帧指标字典
        """
        ego_pose = self.nusc_extractor.extract_ego_pose(sample_token)
        
        # 获取预测和GT数据
        pred_detections = self.sd_extractor.extract_detections(sample_token)
        gt_annotations = self.nusc_extractor.extract_annotations(sample_token)
        
        pred_planning = self.sd_extractor.extract_planning(sample_token)
        gt_planning_list = self._extract_gt_planning(
            sample_token, pred_planning, self.nusc_extractor
        )
        
        pred_map = self.sd_extractor.extract_map_predictions(sample_token)
        
        # 评估检测
        detection_metrics = self.detection_evaluator.evaluate_sample(
            pred_detections, gt_annotations
        )
        
        # 移除详细对象指标，添加统计信息
        per_object_metrics = detection_metrics.pop('per_object_metrics', [])
        detection_metrics['num_tp'] = len([m for m in per_object_metrics if m.get('is_tp', False)])
        detection_metrics['num_fp'] = len([m for m in per_object_metrics if not m.get('is_tp', False)])
        num_gt = len(gt_annotations)
        detection_metrics['num_fn'] = max(0, num_gt - detection_metrics['num_tp'])
        
        # 评估地图
        mapping_metrics = self.mapping_evaluator.evaluate_sample(
            pred_map, static_map
        )
        mapping_metrics.pop('prediction_errors', None)
        
        # 评估运动
        gt_trajectories = self.nusc_extractor.extract_gt_trajectories(sample_token)
        motion_metrics = self.motion_evaluator.evaluate_sample(
            pred_detections, gt_trajectories
        )
        
        # 评估规划
        planning_metrics = self.planning_evaluator.evaluate_sample(
            pred_planning, gt_planning_list, static_map, gt_annotations
        )
        
        # 组合指标
        frame_metrics = {
            'frame_index': frame_index,
            'timestamp': ego_pose['timestamp'],
            'detection': detection_metrics,
            'mapping': mapping_metrics,
            'motion': motion_metrics,
            'planning': planning_metrics
        }
        
        return frame_metrics
    
    def _extract_gt_planning(
        self,
        sample_token: str,
        pred_planning: List,
        nusc_extractor
    ) -> List:
        """
        提取GT规划轨迹
        
        Args:
            sample_token: sample token
            pred_planning: 预测规划
            nusc_extractor: NuScenes提取器
            
        Returns:
            GT规划列表
        """
        gt_planning_list = []
        current_token = sample_token
        for _ in range(min(len(pred_planning), 6)):
            if current_token:
                pose = nusc_extractor.extract_ego_pose(current_token)
                gt_planning_list.append(pose['translation'][:2])
                sample = nusc_extractor.nusc.get('sample', current_token)
                current_token = sample.get('next')
            else:
                break
        return gt_planning_list
    
    def _compute_scene_summary(
        self,
        frame_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算场景级汇总
        
        Args:
            frame_metrics: 每帧指标列表
            
        Returns:
            场景汇总字典
        """
        if not frame_metrics:
            return {}
        
        # 分别提取各任务的指标
        detection_metrics = [fm['detection'] for fm in frame_metrics]
        mapping_metrics = [fm['mapping'] for fm in frame_metrics]
        motion_metrics = [fm['motion'] for fm in frame_metrics]
        planning_metrics = [fm['planning'] for fm in frame_metrics]
        
        # 聚合检测指标
        detection_summary = aggregate_metrics(detection_metrics, [
            'NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'
        ])
        
        # 聚合地图指标
        mapping_summary = self._aggregate_mapping_metrics(mapping_metrics)
        
        # 聚合运动指标
        motion_summary = aggregate_metrics(motion_metrics, [
            'minADE', 'minFDE', 'MR', 'EPA'
        ])
        
        # 聚合规划指标
        planning_summary = aggregate_metrics(planning_metrics, [
            'mean_l2_error', 'max_l2_error'
        ])
        
        # 计算碰撞率
        collision_count = sum(1 for pm in planning_metrics if pm.get('collision_detected', False))
        collision_rate = collision_count / len(planning_metrics) if planning_metrics else 0.0
        
        scene_summary = {
            'detection': detection_summary,
            'mapping': mapping_summary,
            'motion': motion_summary,
            'planning': {
                **planning_summary,
                'collision_rate': collision_rate
            }
        }
        
        return scene_summary
    
    def _aggregate_mapping_metrics(
        self,
        mapping_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        聚合地图指标
        
        适配优化后的mapping评估器结构
        """
        if not mapping_metrics:
            return {
                'AP_ped_crossing': 0.0,
                'AP_divider': 0.0,
                'AP_boundary': 0.0,
                'mAP': 0.0
            }
        
        # 提取各类别AP
        ap_by_class = {
            'ped_crossing': [],
            'divider': [],
            'boundary': []
        }
        map_scores = []
        
        for metric in mapping_metrics:
            ap_by_class_dict = metric.get('AP_by_class', {})
            for cls in ap_by_class.keys():
                if cls in ap_by_class_dict:
                    ap_by_class[cls].append(ap_by_class_dict[cls])
            
            if 'mAP' in metric:
                map_scores.append(metric['mAP'])
        
        # 计算平均AP，保持向后兼容的字段名
        summary = {}
        # ped_crossing -> AP_ped (保持向后兼容)
        if ap_by_class['ped_crossing']:
            summary['AP_ped'] = float(np.mean(ap_by_class['ped_crossing']))
        else:
            summary['AP_ped'] = 0.0
        
        if ap_by_class['divider']:
            summary['AP_divider'] = float(np.mean(ap_by_class['divider']))
        else:
            summary['AP_divider'] = 0.0
        
        if ap_by_class['boundary']:
            summary['AP_boundary'] = float(np.mean(ap_by_class['boundary']))
        else:
            summary['AP_boundary'] = 0.0
        
        if map_scores:
            summary['mAP'] = float(np.mean(map_scores))
        else:
            summary['mAP'] = 0.0
        
        return summary
    
    def save(
        self,
        metrics: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存指标到JSON文件
        
        Args:
            metrics: 指标数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_json(metrics, output_dir, scene_name, 'metrics.json')
