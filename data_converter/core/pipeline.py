"""
场景计算流水线

管理单个场景的所有中间计算结果，实现计算缓存和依赖管理
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from ..utils.matching import ObjectMatcher, MapElementMatcher

logger = logging.getLogger(__name__)


class ScenePipeline:
    """
    场景计算流水线
    
    为单个场景管理所有中间计算结果的缓存，避免重复计算
    """
    
    def __init__(
        self,
        scene_token: str,
        nusc_extractor,
        sd_extractor,
        map_extractor
    ):
        """
        初始化场景流水线
        
        Args:
            scene_token: 场景token
            nusc_extractor: NuScenes数据提取器
            sd_extractor: SparseDrive预测提取器
            map_extractor: 地图提取器
        """
        self.scene_token = scene_token
        self.nusc_extractor = nusc_extractor
        self.sd_extractor = sd_extractor
        self.map_extractor = map_extractor
        
        # 缓存字典
        self._cache = {
            'sample_tokens': None,
            'ego_poses': None,
            'static_map': None,
            'object_matches': {},  # {frame_index: matches}
            'map_matches': {},     # {frame_index: {class_name: matches}}
            'gt_annotations': {},  # {frame_index: annotations}
            'pred_detections': {},  # {frame_index: detections}
            'pred_map': {},        # {frame_index: map}
            'gt_trajectories': {}  # {frame_index: trajectories}
        }
        
        # 初始化匹配器
        self.object_matcher = ObjectMatcher()
        self.map_matcher = MapElementMatcher()
    
    def get_sample_tokens(self) -> List[str]:
        """
        获取或计算场景的sample tokens
        
        Returns:
            sample token列表
        """
        if self._cache['sample_tokens'] is None:
            self._cache['sample_tokens'] = self.nusc_extractor.get_sample_tokens(
                self.scene_token
            )
        return self._cache['sample_tokens']
    
    def get_ego_poses(self) -> List[Dict[str, Any]]:
        """
        获取或计算场景的ego poses
        
        Returns:
            ego pose列表
        """
        if self._cache['ego_poses'] is None:
            self._cache['ego_poses'] = self.nusc_extractor.extract_ego_poses(
                self.scene_token
            )
        return self._cache['ego_poses']
    
    def get_static_map(self) -> Dict[str, List]:
        """
        获取或计算场景的静态地图
        
        Returns:
            静态地图字典
        """
        if self._cache['static_map'] is None:
            ego_poses = self.get_ego_poses()
            self._cache['static_map'] = self.map_extractor.extract_static_map(
                self.scene_token, ego_poses
            )
            logger.debug(f"缓存静态地图 - 场景: {self.scene_token}")
        return self._cache['static_map']
    
    def get_gt_annotations(self, frame_index: int, sample_token: str) -> List[Dict]:
        """
        获取或计算指定帧的GT标注
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            GT标注列表
        """
        if frame_index not in self._cache['gt_annotations']:
            self._cache['gt_annotations'][frame_index] = \
                self.nusc_extractor.extract_annotations(sample_token)
        return self._cache['gt_annotations'][frame_index]
    
    def get_pred_detections(self, frame_index: int, sample_token: str) -> List[Dict]:
        """
        获取或计算指定帧的预测检测
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            预测检测列表
        """
        if frame_index not in self._cache['pred_detections']:
            self._cache['pred_detections'][frame_index] = \
                self.sd_extractor.extract_detections(sample_token)
        return self._cache['pred_detections'][frame_index]
    
    def get_pred_map(self, frame_index: int, sample_token: str) -> List[Dict]:
        """
        获取或计算指定帧的预测地图
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            预测地图列表
        """
        if frame_index not in self._cache['pred_map']:
            self._cache['pred_map'][frame_index] = \
                self.sd_extractor.extract_map_predictions(sample_token)
        return self._cache['pred_map'][frame_index]
    
    def get_gt_trajectories(self, frame_index: int, sample_token: str) -> Dict[str, List]:
        """
        获取或计算指定帧的GT轨迹
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            GT轨迹字典
        """
        if frame_index not in self._cache['gt_trajectories']:
            self._cache['gt_trajectories'][frame_index] = \
                self.nusc_extractor.extract_gt_trajectories(sample_token)
        return self._cache['gt_trajectories'][frame_index]
    
    def get_object_matches(
        self,
        frame_index: int,
        sample_token: str
    ) -> List[Tuple[int, int, float, float]]:
        """
        获取或计算指定帧的对象匹配结果
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            匹配列表 [(gt_idx, pred_idx, iou, distance), ...]
        """
        if frame_index not in self._cache['object_matches']:
            gt_annotations = self.get_gt_annotations(frame_index, sample_token)
            pred_detections = self.get_pred_detections(frame_index, sample_token)
            
            matches = self.object_matcher.match(gt_annotations, pred_detections)
            self._cache['object_matches'][frame_index] = matches
            logger.debug(f"缓存对象匹配 - 帧{frame_index}: {len(matches)}个匹配")
        
        return self._cache['object_matches'][frame_index]
    
    def get_map_matches(
        self,
        frame_index: int,
        sample_token: str,
        class_name: str
    ) -> List[Tuple[int, int, float]]:
        """
        获取或计算指定帧和类别的地图匹配结果
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            class_name: 地图类别名称
            
        Returns:
            匹配列表 [(pred_idx, gt_idx, distance), ...]
        """
        if frame_index not in self._cache['map_matches']:
            self._cache['map_matches'][frame_index] = {}
        
        if class_name not in self._cache['map_matches'][frame_index]:
            pred_map = self.get_pred_map(frame_index, sample_token)
            static_map = self.get_static_map()
            
            matches = self.map_matcher.match(pred_map, static_map, class_name)
            self._cache['map_matches'][frame_index][class_name] = matches
            logger.debug(f"缓存地图匹配 - 帧{frame_index}, {class_name}: {len(matches)}个匹配")
        
        return self._cache['map_matches'][frame_index][class_name]
    
    def get_all_map_matches(
        self,
        frame_index: int,
        sample_token: str
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        获取指定帧的所有地图类别匹配结果
        
        Args:
            frame_index: 帧索引
            sample_token: sample token
            
        Returns:
            {class_name: matches} 字典
        """
        results = {}
        for class_name in ['ped_crossing', 'divider', 'boundary']:
            results[class_name] = self.get_map_matches(frame_index, sample_token, class_name)
        return results
    
    def clear_frame_cache(self, frame_index: int):
        """
        清除指定帧的缓存（释放内存）
        
        Args:
            frame_index: 帧索引
        """
        self._cache['object_matches'].pop(frame_index, None)
        self._cache['map_matches'].pop(frame_index, None)
        self._cache['gt_annotations'].pop(frame_index, None)
        self._cache['pred_detections'].pop(frame_index, None)
        self._cache['pred_map'].pop(frame_index, None)
        self._cache['gt_trajectories'].pop(frame_index, None)
    
    def clear_all_cache(self):
        """
        清除所有缓存
        """
        self._cache = {
            'sample_tokens': None,
            'ego_poses': None,
            'static_map': None,
            'object_matches': {},
            'map_matches': {},
            'gt_annotations': {},
            'pred_detections': {},
            'pred_map': {},
            'gt_trajectories': {}
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        return {
            'scene_token': self.scene_token,
            'has_sample_tokens': self._cache['sample_tokens'] is not None,
            'has_ego_poses': self._cache['ego_poses'] is not None,
            'has_static_map': self._cache['static_map'] is not None,
            'cached_frames': {
                'object_matches': len(self._cache['object_matches']),
                'map_matches': len(self._cache['map_matches']),
                'gt_annotations': len(self._cache['gt_annotations']),
                'pred_detections': len(self._cache['pred_detections']),
                'pred_map': len(self._cache['pred_map']),
                'gt_trajectories': len(self._cache['gt_trajectories'])
            }
        }


