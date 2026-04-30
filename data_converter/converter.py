"""
数据转换主协调器

协调所有builders，生成完整的数据文件

核心方法：
- convert_scene: 转换单个场景
- convert_scenes: 批量转换多个场景（默认多线程）
"""

import logging
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core.nuscenes_extractor import NuScenesExtractor
from .core.sparsedrive_extractor import SparseDriveExtractor
from .core.map_extractor import MapExtractor
from .core.pipeline import ScenePipeline

from .builders.metadata_builder import MetadataBuilder
from .builders.map_builder import MapBuilder
from .builders.basemap_builder import BasemapBuilder
from .builders.gt_stream_builder import GtStreamBuilder
from .builders.prediction_stream_builder import PredictionStreamBuilder
from .builders.metrics_builder import MetricsBuilder
from .builders.associations_builder import AssociationsBuilder
from .builders.map_coloring_builder import MapColoringBuilder
from .builders.camera_stream_builder import CameraStreamBuilder
from .builders.scene_index_builder import SceneIndexBuilder
from .builders.global_statistics_builder import GlobalStatisticsBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConverter:
    """
    数据转换器
    
    协调所有提取器和构建器，生成前端所需的数据文件
    使用ScenePipeline管理计算流水线，避免重复计算
    默认使用多线程批量转换
    """

    def __init__(
            self,
            nuscenes_dataroot: str,
            sparsedrive_prediction: str,
            nuscenes_version: str = 'v1.0-trainval'
    ):
        """
        初始化转换器
        
        Args:
            nuscenes_dataroot: NuScenes数据根目录
            sparsedrive_prediction: SparseDrive预测文件路径
            nuscenes_version: NuScenes数据集版本
        """
        self.nuscenes_dataroot = nuscenes_dataroot
        self.sparsedrive_prediction = sparsedrive_prediction
        self.nuscenes_version = nuscenes_version

        # 初始化提取器
        self.nusc_extractor = NuScenesExtractor(nuscenes_dataroot, nuscenes_version)
        self.sd_extractor = SparseDriveExtractor(sparsedrive_prediction, self.nusc_extractor)
        self.map_extractor = MapExtractor(self.nusc_extractor.nusc)

        # 构建场景名称到token的映射
        self._scene_name_to_token = {
            scene['name']: scene['token']
            for scene in self.nusc_extractor.nusc.scene
        }

        # 初始化构建器
        self.metadata_builder = MetadataBuilder(self.nusc_extractor, self.map_extractor)
        self.map_builder = MapBuilder(self.nusc_extractor, self.map_extractor)
        self.basemap_builder = BasemapBuilder(self.nusc_extractor, self.map_extractor)
        self.gt_stream_builder = GtStreamBuilder(self.nusc_extractor)
        self.pred_stream_builder = PredictionStreamBuilder(self.nusc_extractor, self.sd_extractor)
        self.metrics_builder = MetricsBuilder(self.nusc_extractor, self.sd_extractor, self.map_extractor)
        self.associations_builder = AssociationsBuilder(self.nusc_extractor, self.sd_extractor, self.map_extractor)
        self.map_coloring_builder = MapColoringBuilder(self.nusc_extractor, self.sd_extractor, self.map_extractor)
        self.camera_stream_builder = CameraStreamBuilder(self.nusc_extractor)
        self.scene_index_builder = SceneIndexBuilder(self.nusc_extractor, self.map_extractor)

        logger.info(f"数据转换器初始化完成 (场景数: {len(self._scene_name_to_token)})")

    def convert_scene(self, scene_name: str, output_dir: str) -> dict:
        """
        转换单个场景
        
        Args:
            scene_name: 场景名称
            output_dir: 输出目录
            
        Returns:
            转换结果
        """
        scene_token = self._scene_name_to_token.get(scene_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"转换场景: {scene_name}")

        # 创建场景pipeline
        pipeline = ScenePipeline(
            scene_token,
            self.nusc_extractor,
            self.sd_extractor,
            self.map_extractor
        )

        files = {}

        # 构建所有数据文件
        metadata = self.metadata_builder.build(pipeline)
        files['metadata'] = str(self.metadata_builder.save(metadata, output_path, scene_name))
        
        # static_map = self.map_builder.build(pipeline)
        # files['static_map'] = str(self.map_builder.save(static_map, output_path, scene_name))

        # gt_stream = self.gt_stream_builder.build(pipeline)
        # files['gt_stream'] = str(self.gt_stream_builder.save(gt_stream, output_path, scene_name))

        # pred_stream = self.pred_stream_builder.build(pipeline)
        # files['prediction_stream'] = str(self.pred_stream_builder.save(pred_stream, output_path, scene_name))

        # metrics = self.metrics_builder.build(pipeline)
        # files['metrics'] = str(self.metrics_builder.save(metrics, output_path, scene_name))

        # associations = self.associations_builder.build(pipeline)
        # files['associations'] = str(self.associations_builder.save(associations, output_path, scene_name))

        # map_coloring = self.map_coloring_builder.build(pipeline)
        # files['map_coloring'] = str(self.map_coloring_builder.save(map_coloring, output_path, scene_name))

        from .config import config
        
        # 生成栅格底图
        if config.GENERATE_BASEMAP:
            basemap_data = self.basemap_builder.build(pipeline)
            basemap_files = self.basemap_builder.save(basemap_data, output_path, scene_name)
            files['basemap_image'] = str(basemap_files['image'])
            files['basemap_metadata'] = str(basemap_files['metadata'])
        
        if config.GENERATE_CAMERA_STREAM:
            camera_stream = self.camera_stream_builder.build(pipeline, output_path, scene_name)
            files['camera_stream'] = str(self.camera_stream_builder.save(camera_stream, output_path, scene_name))

        pipeline.clear_all_cache()

        logger.info(f"完成: {scene_name} -> {output_path / scene_name}")
        return {
            'scene_name': scene_name,
            'files': files,
            'metadata': metadata  # 返回metadata用于全局统计
        }

    def convert_scenes(
            self,
            scene_names: List[str],
            output_dir: str,
            max_workers: int = 4,
            generate_index: bool = True
    ) -> dict:
        """
        批量转换多个场景（默认多线程）

        Args:
            scene_names: 场景名称列表
            output_dir: 输出目录
            max_workers: 最大线程数
            generate_index: 是否生成索引文件

        Returns:
            {
                'scenes': List[dict],                  # 各场景转换结果
                'index_file': str,                     # 索引文件路径
                'global_statistics_file': str          # 全局统计文件路径
            }
        """
        # 过滤无效场景
        valid_scenes = [name for name in scene_names if name in self._scene_name_to_token]
        invalid_count = len(scene_names) - len(valid_scenes)

        if invalid_count > 0:
            logger.warning(f"跳过 {invalid_count} 个不存在的场景")

        logger.info(f"批量转换 {len(valid_scenes)} 个场景 (多线程, {max_workers}线程)")

        results = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scene = {
                executor.submit(self.convert_scene, scene_name, output_dir): scene_name
                for scene_name in valid_scenes
            }
            for future in as_completed(future_to_scene):
                completed_count += 1
                result = future.result()
                results.append(result)
                logger.info(f"[{completed_count}/{len(valid_scenes)}] 完成: {result['scene_name']}")

        logger.info(f"批量转换完成: {len(results)} 个场景")

        # 生成全局统计文件
        global_statistics_file = None
        logger.info("生成全局对象统计文件...")
        scene_metadata_list = [result.get('metadata') for result in results if result.get('metadata')]
        if scene_metadata_list:
            global_stats_builder = GlobalStatisticsBuilder()
            global_statistics = global_stats_builder.build(scene_metadata_list)
            output_path = Path(output_dir)
            global_statistics_file = global_stats_builder.save(global_statistics, output_path)
            logger.info(f"全局统计文件已生成: {global_statistics_file}")

        # 生成索引文件
        index_file = None

        if generate_index:
            from .config import config
            output_path = Path(output_dir)

            if config.GENERATE_SCENE_INDEX:
                logger.info("生成场景索引文件...")
                scene_tokens = [self._scene_name_to_token[name] for name in valid_scenes]
                index_data = self.scene_index_builder.build_index(
                    scene_tokens,
                    output_path,
                    # config.THUMBNAIL_SAMPLE_RATE
                )
                index_file = self.scene_index_builder.save(index_data, output_path)
                logger.info(f"场景索引已生成: {index_file}")

        return {
            'scenes': results,
            'index_file': str(index_file) if index_file else None,
            'global_statistics_file': str(global_statistics_file) if global_statistics_file else None
        }
