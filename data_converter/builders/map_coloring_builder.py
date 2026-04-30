"""
地图着色构建器（优化版本）

为GT地图元素添加基于预测误差的着色信息
用于可视化不同道路位置的模型预测表现

优化说明：
- 使用统一的地图匹配器
- 复用pipeline中的地图匹配结果
- 消除重复的插值和距离计算函数
"""

import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
from scipy.spatial.distance import cdist

from .base_builder import BaseBuilder
from ..core.pipeline import ScenePipeline
from ..utils.distance import interpolate_line


class MapColoringBuilder(BaseBuilder):
    """
    地图着色构建器
    
    为GT地图元素的每个点计算预测误差，生成着色数据
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
    
    def build(self, scene_token_or_pipeline: Union[str, ScenePipeline]) -> Dict[str, Any]:
        """
        构建地图着色数据
        
        Args:
            scene_token_or_pipeline: 场景token或ScenePipeline
            
        Returns:
            着色数据字典
        """
        # 检测是pipeline还是传统模式
        if isinstance(scene_token_or_pipeline, ScenePipeline):
            return self._build_from_pipeline(scene_token_or_pipeline)
        else:
            return self._build_traditional(scene_token_or_pipeline)
    
    def _build_from_pipeline(self, pipeline: ScenePipeline) -> Dict[str, Any]:
        """
        从pipeline构建地图着色（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            
        Returns:
            着色数据字典
        """
        sample_tokens = pipeline.get_sample_tokens()
        gt_map = pipeline.get_static_map()
        
        # 构建GT到预测的映射（复用pipeline的匹配结果）
        gt_to_predictions = self._build_gt_to_predictions_from_pipeline(
            pipeline, sample_tokens, gt_map
        )
        
        # 为每个类别处理
        colored_elements = {}
        for category in ['divider', 'boundary', 'ped_crossing']:
            colored_elements[category] = self._process_category(
                gt_map[category],
                gt_to_predictions,
                category,
                sample_tokens
            )
        
        return {
            'scene_token': pipeline.scene_token,
            'colored_elements': colored_elements
        }
    
    def _build_traditional(self, scene_token: str) -> Dict[str, Any]:
        """
        传统方式构建地图着色（向后兼容）
        
        Args:
            scene_token: 场景token
            
        Returns:
            着色数据字典
        """
        sample_tokens = self.nusc_extractor.get_sample_tokens(scene_token)
        ego_poses = self.nusc_extractor.extract_ego_poses(scene_token)
        gt_map = self.map_extractor.extract_static_map(scene_token, ego_poses)
        
        # 构建GT到预测的映射（使用统一匹配器）
        gt_to_predictions = self._build_gt_to_predictions_traditional(
            sample_tokens, gt_map
        )
        
        # 为每个类别处理
        colored_elements = {}
        for category in ['divider', 'boundary', 'ped_crossing']:
            colored_elements[category] = self._process_category(
                gt_map[category],
                gt_to_predictions,
                category,
                sample_tokens
            )
        
        return {
            'scene_token': scene_token,
            'colored_elements': colored_elements
        }
    
    def _build_gt_to_predictions_from_pipeline(
        self,
        pipeline: ScenePipeline,
        sample_tokens: List[str],
        gt_map: Dict[str, List]
    ) -> Dict[str, List[Dict]]:
        """
        从pipeline构建GT元素到预测的映射（优化模式）
        
        Args:
            pipeline: 场景计算流水线
            sample_tokens: sample token列表
            gt_map: GT地图
            
        Returns:
            {gt_element_id: [预测信息列表]}
        """
        gt_to_predictions = {}
        
        # 遍历每一帧，复用pipeline中的匹配结果
        for frame_index, sample_token in enumerate(sample_tokens):
            if not pipeline.sd_extractor.has_prediction(sample_token):
                continue
            
            pred_map = pipeline.get_pred_map(frame_index, sample_token)
            if not pred_map:
                continue
            
            # 获取所有类别的匹配结果（复用pipeline缓存）
            all_matches = pipeline.get_all_map_matches(frame_index, sample_token)
            
            # 处理每个类别的匹配
            for class_name, matches in all_matches.items():
                for pred_idx, gt_idx, chamfer_dist in matches:
                    gt_element_id = f"{class_name}:{gt_idx}"
                    
                    if gt_element_id not in gt_to_predictions:
                        gt_to_predictions[gt_element_id] = []
                    
                    pred_item = pred_map[pred_idx]
                    gt_to_predictions[gt_element_id].append({
                        'frame_index': frame_index,
                        'sample_token': sample_token,
                        'pred_index': pred_idx,
                        'pred_vectors': pred_item.get('vectors', []),
                        'score': pred_item.get('score', 0.0),
                        'chamfer_distance': chamfer_dist
                    })
        
        return gt_to_predictions
    
    def _build_gt_to_predictions_traditional(
        self,
        sample_tokens: List[str],
        gt_map: Dict[str, List]
    ) -> Dict[str, List[Dict]]:
        """
        传统方式构建GT元素到预测的映射
        
        Args:
            sample_tokens: sample token列表
            gt_map: GT地图
            
        Returns:
            {gt_element_id: [预测信息列表]}
        """
        gt_to_predictions = {}
        
        # 使用统一的地图匹配器
        from ..utils.matching import MapElementMatcher
        map_matcher = MapElementMatcher()
        
        # 遍历每一帧进行匹配
        for frame_index, sample_token in enumerate(sample_tokens):
            if not self.sd_extractor.has_prediction(sample_token):
                continue
            
            pred_map = self.sd_extractor.extract_map_predictions(sample_token)
            if not pred_map:
                continue
            
            # 按类别匹配
            for class_name in ["ped_crossing", "divider", "boundary"]:
                matches = map_matcher.match(pred_map, gt_map, class_name)
                
                # 更新映射
                for pred_idx, gt_idx, chamfer_dist in matches:
                    gt_element_id = f"{class_name}:{gt_idx}"
                    
                    if gt_element_id not in gt_to_predictions:
                        gt_to_predictions[gt_element_id] = []
                    
                    pred_item = pred_map[pred_idx]
                    gt_to_predictions[gt_element_id].append({
                        'frame_index': frame_index,
                        'sample_token': sample_token,
                        'pred_index': pred_idx,
                        'pred_vectors': pred_item.get('vectors', []),
                        'score': pred_item.get('score', 0.0),
                        'chamfer_distance': chamfer_dist
                    })
        
        return gt_to_predictions
    
    def _process_category(
        self,
        gt_elements: List,
        gt_to_predictions: Dict[str, List[Dict]],
        category: str,
        sample_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        处理单个类别的GT元素
        
        Args:
            gt_elements: GT元素列表
            gt_to_predictions: GT到预测的映射
            category: 类别名称
            sample_tokens: sample token列表
            
        Returns:
            着色后的元素列表
        """
        colored_elements = []
        
        for gt_idx, gt_element in enumerate(gt_elements):
            gt_element_id = f"{category}:{gt_idx}"
            predictions = gt_to_predictions.get(gt_element_id, [])
            
            if not predictions:
                # 未被预测到的GT元素
                colored_elements.append({
                    'element_id': gt_element_id,
                    'coordinates': gt_element,
                    'point_errors': [0.0] * len(gt_element),
                    'avg_error': 0.0,
                    'max_error': 0.0,
                    'min_error': 0.0,
                    'prediction_count': 0,
                    'coverage_ratio': 0.0
                })
                continue
            
            # 计算误差和统计信息
            error_info = self._compute_error_info(gt_element, predictions)
            
            colored_elements.append({
                'element_id': gt_element_id,
                'coordinates': gt_element,
                'point_errors': error_info['point_errors'],
                'avg_error': error_info['avg_error'],
                'max_error': error_info['max_error'],
                'min_error': error_info['min_error'],
                'prediction_count': len(predictions),
                'coverage_ratio': error_info['coverage_ratio']
            })
        
        return colored_elements
    
    def _compute_error_info(
        self,
        gt_element: List,
        predictions: List[Dict]
    ) -> Dict[str, Any]:
        """
        计算GT元素的误差信息
        
        Args:
            gt_element: GT元素的点列表
            predictions: 预测信息列表
            
        Returns:
            误差信息字典
        """
        gt_points = np.array(gt_element)
        
        # 收集所有预测向量
        pred_vectors = []
        for pred in predictions:
            vectors = pred.get('pred_vectors', [])
            if vectors and len(vectors) > 0:
                pred_vectors.append(np.array(vectors))
        
        if not pred_vectors:
            # 没有有效预测
            return {
                'point_errors': [0.0] * len(gt_points),
                'avg_error': 0.0,
                'max_error': 0.0,
                'min_error': 0.0,
                'coverage_ratio': 0.0
            }
        
        # 计算每个GT点到所有预测的最小距离
        point_errors = []
        coverage_threshold = 3.0  # 小于3米认为被覆盖
        covered_points = 0
        
        for gt_point in gt_points:
            min_dist = float('inf')
            for pred_vec in pred_vectors:
                dists = np.linalg.norm(pred_vec - gt_point, axis=1)
                min_dist = min(min_dist, dists.min())
            
            point_errors.append(float(min_dist))
            
            if min_dist < coverage_threshold:
                covered_points += 1
        
        # 计算统计信息
        point_errors_array = np.array(point_errors)
        coverage_ratio = covered_points / len(gt_points) if len(gt_points) > 0 else 0.0
        
        return {
            'point_errors': [round(e, 2) for e in point_errors],
            'avg_error': round(float(point_errors_array.mean()), 2),
            'max_error': round(float(point_errors_array.max()), 2),
            'min_error': round(float(point_errors_array.min()), 2),
            'coverage_ratio': round(coverage_ratio, 2)
        }
    
    def save(
        self,
        coloring_data: Dict[str, Any],
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        保存着色数据到JSON文件
        
        Args:
            coloring_data: 着色数据
            output_dir: 输出目录
            scene_name: 场景名称
            
        Returns:
            保存的文件路径
        """
        return self.save_json(coloring_data, output_dir, scene_name, 'mapping_color.json')
