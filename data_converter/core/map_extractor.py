"""
地图数据提取器

基于temp/services/map_service.py重构
提取NuScenes静态地图数据
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from shapely.geometry import Polygon, LineString, box
from shapely import ops
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely.geometry.base import BaseGeometry

from ..config import config

logger = logging.getLogger(__name__)

# 过滤 Shapely 的 self-intersection INFO 警告
logging.getLogger('shapely.geos').setLevel(logging.WARNING)


# 支持的地图位置
LOCATIONS = [
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
]


class GeometryProcessor:
    """几何体处理器 - 提供增强的几何修复和处理功能"""

    @staticmethod
    def repair_geometry(geom: BaseGeometry) -> Optional[BaseGeometry]:
        """修复无效的几何对象，使用多重修复策略"""
        if not geom or geom.is_empty:
            return None
            
        try:
            if geom.is_valid:
                return geom
                
            # 策略1: buffer(0)
            try:
                repaired = geom.buffer(0)
                if repaired and not repaired.is_empty and repaired.is_valid:
                    return repaired
            except Exception:
                pass
                
            # 策略2: make_valid (Shapely 2.0+)
            if hasattr(geom, 'make_valid'):
                try:
                    valid_geom = geom.make_valid()
                    if valid_geom and not valid_geom.is_empty and valid_geom.is_valid:
                        return valid_geom
                except Exception:
                    pass
                    
            # 策略3: 对于多边形，重新构建外环
            if geom.geom_type == 'Polygon':
                try:
                    exterior_coords = list(geom.exterior.coords)
                    if len(exterior_coords) >= 4:
                        new_polygon = Polygon(exterior_coords)
                        if new_polygon.is_valid:
                            return new_polygon
                        else:
                            buffered = new_polygon.buffer(0)
                            if buffered and buffered.is_valid:
                                return buffered
                except Exception:
                    pass
                    
            # 策略4: 简化几何对象
            try:
                simplified = geom.simplify(0.1, preserve_topology=True)
                if simplified and not simplified.is_empty and simplified.is_valid:
                    return simplified
            except Exception:
                pass
                
            # 策略5: 大tolerance简化
            try:
                simplified = geom.simplify(1.0, preserve_topology=False)
                if simplified and not simplified.is_empty and simplified.is_valid:
                    return simplified
            except Exception:
                pass
                
            return None
            
        except Exception:
            return None

    @staticmethod
    def safe_buffer(geom: BaseGeometry, distance: float) -> Optional[BaseGeometry]:
        """安全的buffer操作"""
        try:
            repaired = GeometryProcessor.repair_geometry(geom)
            if not repaired:
                return None
                
            buffered = repaired.buffer(distance)
            return GeometryProcessor.repair_geometry(buffered)
            
        except Exception:
            return None

    @staticmethod
    def normalize_geometry(geom: BaseGeometry) -> List[BaseGeometry]:
        """将任何几何体标准化为有效的几何体列表"""
        if not geom or geom.is_empty:
            return []

        repaired = GeometryProcessor.repair_geometry(geom)
        if not repaired:
            return []

        if repaired.geom_type in ['Polygon', 'LineString']:
            return [repaired] if repaired.is_valid else []

        if repaired.geom_type in ['MultiPolygon', 'MultiLineString']:
            return [g for g in repaired.geoms if g.is_valid and not g.is_empty]

        if repaired.geom_type == 'GeometryCollection':
            result = []
            for g in repaired.geoms:
                result.extend(GeometryProcessor.normalize_geometry(g))
            return result

        return []

    @staticmethod
    def filter_by_threshold(
        geometries: List[BaseGeometry],
        min_length: float = 1.0,
        min_area: float = 5.0
    ) -> List[BaseGeometry]:
        """过滤小几何体"""
        result = []
        for geom in geometries:
            if geom.geom_type in ['LineString', 'MultiLineString']:
                if geom.length > min_length:
                    result.append(geom)
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                if geom.area > min_area:
                    result.append(geom)
            else:
                result.append(geom)
        return result

    @staticmethod
    def validate_and_repair_collection(geometries: List[BaseGeometry]) -> List[BaseGeometry]:
        """批量验证和修复几何对象集合"""
        valid_geometries = []
        for geom in geometries:
            if not geom or geom.is_empty:
                continue
                
            if geom.is_valid:
                valid_geometries.append(geom)
            else:
                repaired = GeometryProcessor.repair_geometry(geom)
                if repaired:
                    valid_geometries.append(repaired)
                    
        return valid_geometries

    @staticmethod
    def safe_unary_union(geometries: List[BaseGeometry]) -> Optional[BaseGeometry]:
        """安全的unary_union操作"""
        if not geometries:
            return None
            
        try:
            valid_geometries = GeometryProcessor.validate_and_repair_collection(geometries)
            if not valid_geometries:
                return None
                
            if len(valid_geometries) == 1:
                return valid_geometries[0]
                
            unified = ops.unary_union(valid_geometries)
            
            repaired_unified = GeometryProcessor.repair_geometry(unified)
            if repaired_unified:
                return repaired_unified
            else:
                return None
                
        except Exception:
            try:
                if valid_geometries:
                    largest = max(valid_geometries, key=lambda g: g.area if hasattr(g, 'area') else g.length)
                    return largest
            except Exception:
                pass
            return None


class MapExtractor:
    """
    地图提取器
    
    提取NuScenes静态地图数据，包括车道线、道路边界、人行横道等
    """
    
    def __init__(self, nuscenes: NuScenes):
        """
        初始化提取器
        
        Args:
            nuscenes: NuScenes实例
        """
        self.nusc = nuscenes
        self.dataroot = nuscenes.dataroot
        self.map_explorers = self._initialize_map_explorers()
    
    def _initialize_map_explorers(self) -> Dict[str, NuScenesMapExplorer]:
        """
        初始化所有地图探索器
        
        Returns:
            {location: NuScenesMapExplorer}
        """
        explorers = {}
        for location in LOCATIONS:
            nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=location)
            explorers[location] = NuScenesMapExplorer(nusc_map)
        return explorers
    
    def _get_map_explorer(self, scene_token: str) -> NuScenesMapExplorer:
        """
        获取场景对应的地图探索器
        
        Args:
            scene_token: 场景token
            
        Returns:
            NuScenesMapExplorer实例
        """
        scene = self.nusc.get('scene', scene_token)
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']
        return self.map_explorers[location]
    
    def extract_static_map(
        self,
        scene_token: str,
        ego_poses: List[Dict[str, Any]],
        map_margin: float = None,
    ) -> Dict[str, List]:
        """
        提取场景的静态地图（全局坐标系）
        
        Args:
            scene_token: 场景token
            ego_poses: 自车位姿列表，用于确定地图范围
            map_margin: 地图边界扩展（米），None表示使用配置的默认值

        Returns:
            地图字典，包含dividers, boundaries, ped_crossings, drivable_area
            
        原理：
            1. 根据ego_poses计算地图覆盖范围
            2. 提取范围内的所有地图元素
            3. 返回全局坐标系的地图数据
        """
        if map_margin is None:
            map_margin = config.MAP_MARGIN
        
        # 计算地图范围
        map_range = self._compute_map_range(ego_poses, map_margin)
        
        # 获取地图探索器
        map_explorer = self._get_map_explorer(scene_token)
        
        # 提取地图元素
        static_map = {
            'divider': self._extract_dividers(map_explorer, map_range),
            'boundary': self._extract_boundaries(map_explorer, map_range),
            'ped_crossing': self._extract_ped_crossings(map_explorer, map_range),
            'drivable_area':self._extract_drivable_area(map_explorer, map_range)
        }
        

        
        return static_map
    
    def _compute_map_range(
        self,
        ego_poses: List[Dict[str, Any]],
        map_margin: float
    ) -> Polygon:
        """计算地图覆盖范围"""
        patch_boxes = []
        for ego_pose in ego_poses:
            translation = ego_pose['translation']
            yaw_deg = ego_pose['yaw'] / np.pi * 180
            
            patch = self._create_patch_polygon(
                translation[0], translation[1],
                config.DEFAULT_PATCH_SIZE[0], config.DEFAULT_PATCH_SIZE[1],
                yaw_deg
            )
            patch_boxes.append(patch)
        
        valid_patches = [p for p in patch_boxes if p.is_valid and not p.is_empty]
        if not valid_patches:
            return box(0, 0, 100, 100)
        
        unified = ops.unary_union(valid_patches)
        
        if unified.geom_type == 'MultiPolygon':
            unified = max(unified.geoms, key=lambda p: p.area)
        
        if unified.is_valid:
            return unified.buffer(map_margin)
        else:
            repaired = GeometryProcessor.repair_geometry(unified)
            if repaired:
                return repaired.buffer(map_margin)
            else:
                return box(0, 0, 100, 100)
    
    def _create_patch_polygon(
        self,
        x: float,
        y: float,
        patch_h: float,
        patch_w: float,
        angle_deg: float = 0.0
    ) -> Polygon:
        """创建patch多边形"""
        x_min = x - patch_w / 2.0
        y_min = y - patch_h / 2.0
        x_max = x + patch_w / 2.0
        y_max = y + patch_h / 2.0
        
        patch = box(x_min, y_min, x_max, y_max)
        
        if angle_deg != 0.0:
            from shapely import affinity
            patch = affinity.rotate(patch, angle_deg, origin=(x, y), use_radians=False)
        
        return patch
    
    def _validate_geometry(self, geom: BaseGeometry) -> BaseGeometry:
        """验证并修复几何对象"""
        return GeometryProcessor.repair_geometry(geom)
    
    def _extract_dividers(
        self,
        map_explorer: NuScenesMapExplorer,
        map_range: Polygon
    ) -> List[List[List[float]]]:
        """提取车道分隔线"""
        expanded_range = GeometryProcessor.safe_buffer(map_range, config.MAP_EXTRACTION_BUFFER)
        if not expanded_range:
            expanded_range = map_range
        
        dividers = []
        map_api = map_explorer.map_api
        
        for layer_name in ['lane_divider', 'road_divider']:
            if layer_name not in map_api.non_geometric_line_layers:
                continue
            
            records = getattr(map_api, layer_name, [])
            for record in records:
                line = map_api.extract_line(record.get('line_token'))
                line = GeometryProcessor.repair_geometry(line)
                if not line:
                    continue
                
                if line.intersects(expanded_range):
                    clipped = self._safe_intersection(line, map_range)
                    if clipped:
                        normalized = GeometryProcessor.normalize_geometry(clipped)
                        filtered = GeometryProcessor.filter_by_threshold(normalized, min_length=1.0)
                        for geom in filtered:
                            coords = self._geometry_to_coords(geom)
                            if coords:
                                dividers.append(coords)
        
        return dividers
    
    def _extract_boundaries(
        self,
        map_explorer: NuScenesMapExplorer,
        map_range: Polygon
    ) -> List[List[List[float]]]:
        """提取道路边界"""
        expanded_range = GeometryProcessor.safe_buffer(map_range, config.MAP_EXTRACTION_BUFFER)
        if not expanded_range:
            expanded_range = map_range
            
        map_api = map_explorer.map_api
        
        drivable_areas = []
        for layer_name in ['road_segment', 'lane']:
            if layer_name not in map_api.non_geometric_polygon_layers:
                continue
            
            records = getattr(map_api, layer_name, [])
            for record in records:
                polygon_tokens = []
                if 'polygon_tokens' in record:
                    polygon_tokens.extend(record['polygon_tokens'])
                elif 'polygon_token' in record:
                    polygon_tokens.append(record['polygon_token'])
                
                for token in polygon_tokens:
                    polygon = map_api.extract_polygon(token)
                    polygon = GeometryProcessor.repair_geometry(polygon)
                    if polygon and polygon.intersects(expanded_range):
                        drivable_areas.append(polygon)
        
        if not drivable_areas:
            return []
        
        unified = GeometryProcessor.safe_unary_union(drivable_areas)
        if not unified or unified.is_empty:
            return []
        
        boundaries = []
        shrunk_range = GeometryProcessor.safe_buffer(map_range, -config.MAP_BOUNDARY_SHRINK)
        if not shrunk_range:
            return []
        
        polygons = []
        if unified.geom_type == 'Polygon':
            polygons = [unified]
        elif unified.geom_type == 'MultiPolygon':
            polygons = list(unified.geoms)
        
        for poly in polygons:
            ext_line = LineString(poly.exterior.coords)
            clipped = self._safe_intersection(ext_line, shrunk_range)
            if clipped:
                normalized = GeometryProcessor.normalize_geometry(clipped)
                filtered = GeometryProcessor.filter_by_threshold(normalized, min_length=5.0)
                for geom in filtered:
                    coords = self._geometry_to_coords(geom)
                    if coords:
                        boundaries.append(coords)
            
            for interior in poly.interiors:
                int_line = LineString(interior.coords)
                clipped = self._safe_intersection(int_line, shrunk_range)
                if clipped:
                    normalized = GeometryProcessor.normalize_geometry(clipped)
                    filtered = GeometryProcessor.filter_by_threshold(normalized, min_length=5.0)
                    for geom in filtered:
                        coords = self._geometry_to_coords(geom)
                        if coords:
                            boundaries.append(coords)
        
        return boundaries
    
    def _extract_ped_crossings(
        self,
        map_explorer: NuScenesMapExplorer,
        map_range: Polygon
    ) -> List[List[List[float]]]:
        """提取人行横道"""
        map_api = map_explorer.map_api
        
        crossings = []
        if 'ped_crossing' not in map_api.non_geometric_polygon_layers:
            return crossings
        
        records = getattr(map_api, 'ped_crossing', [])
        for record in records:
            polygon_token = record.get('polygon_token')
            if not polygon_token:
                continue
            
            polygon = map_api.extract_polygon(polygon_token)
            polygon = GeometryProcessor.repair_geometry(polygon)
            if not polygon:
                continue
            
            if polygon.intersects(map_range):
                exterior = LineString(polygon.exterior.coords)
                
                if map_range.contains(polygon):
                    coords = list(exterior.coords)
                    if coords:
                        crossings.append(coords)
                else:
                    shrunk_range = GeometryProcessor.safe_buffer(map_range, -1.0)
                    if not shrunk_range:
                        continue
                    clipped = self._safe_intersection(exterior, shrunk_range)
                    if clipped:
                        normalized = GeometryProcessor.normalize_geometry(clipped)
                        filtered = GeometryProcessor.filter_by_threshold(normalized, min_length=1.0)
                        for geom in filtered:
                            coords = self._geometry_to_coords(geom)
                            if coords:
                                crossings.append(coords)
        
        return crossings
    
    def _extract_drivable_area(
        self,
        map_explorer: NuScenesMapExplorer,
        map_range: Polygon
    ) -> List[List[List[float]]]:
        """提取可驾驶区域"""
        expanded_range = GeometryProcessor.safe_buffer(map_range, config.MAP_EXTRACTION_BUFFER / 2.0)
        if not expanded_range:
            expanded_range = map_range
            
        map_api = map_explorer.map_api
        
        drivable_polygons = []
        for layer_name in ['road_segment', 'lane']:
            if layer_name not in map_api.non_geometric_polygon_layers:
                continue
            
            records = getattr(map_api, layer_name, [])
            for record in records:
                polygon_tokens = []
                if 'polygon_tokens' in record:
                    polygon_tokens.extend(record['polygon_tokens'])
                elif 'polygon_token' in record:
                    polygon_tokens.append(record['polygon_token'])
                
                for token in polygon_tokens:
                    polygon = map_api.extract_polygon(token)
                    polygon = GeometryProcessor.repair_geometry(polygon)
                    if polygon and polygon.intersects(expanded_range):
                        drivable_polygons.append(polygon)
        
        if not drivable_polygons:
            return []
        
        unified = GeometryProcessor.safe_unary_union(drivable_polygons)
        if not unified or unified.is_empty:
            return []
        
        clipped_areas = []
        
        polygons_to_process = []
        if unified.geom_type == 'Polygon':
            polygons_to_process = [unified]
        elif unified.geom_type == 'MultiPolygon':
            polygons_to_process = list(unified.geoms)
        
        for poly in polygons_to_process:
            if map_range.contains(poly):
                coords = self._polygon_to_coords(poly)
                if coords:
                    clipped_areas.append(coords)
            elif poly.intersects(map_range):
                clipped = self._safe_intersection(poly, map_range)
                if clipped:
                    if clipped.geom_type == 'Polygon':
                        coords = self._polygon_to_coords(clipped)
                        if coords:
                            clipped_areas.append(coords)
                    elif clipped.geom_type == 'MultiPolygon':
                        for sub_poly in clipped.geoms:
                            coords = self._polygon_to_coords(sub_poly)
                            if coords:
                                clipped_areas.append(coords)
        
        return clipped_areas
    
    def _polygon_to_coords(self, polygon: Polygon) -> List[List[float]]:
        """将Polygon转换为坐标列表（只提取外轮廓）"""
        if not polygon or polygon.is_empty:
            return []
        
        try:
            coords = list(polygon.exterior.coords)
            
            if polygon.area < 10.0:
                return []
            return coords
        except Exception:
            return []
    
    def _safe_intersection(
        self,
        geom1: BaseGeometry,
        geom2: BaseGeometry
    ) -> BaseGeometry:
        """安全的几何交集操作"""
        try:
            if not geom1 or geom1.is_empty or not geom2 or geom2.is_empty:
                return None
                
            repaired_geom1 = GeometryProcessor.repair_geometry(geom1)
            repaired_geom2 = GeometryProcessor.repair_geometry(geom2)
            
            if not repaired_geom1 or not repaired_geom2:
                return None
                
            if not repaired_geom1.intersects(repaired_geom2):
                return None
                
            try:
                intersection = repaired_geom1.intersection(repaired_geom2)
                if intersection and not intersection.is_empty:
                    repaired_intersection = GeometryProcessor.repair_geometry(intersection)
                    if repaired_intersection:
                        return repaired_intersection
                    
            except Exception:
                try:
                    buffered_geom1 = repaired_geom1.buffer(0.001)
                    buffered_geom2 = repaired_geom2.buffer(0.001)
                    if buffered_geom1.is_valid and buffered_geom2.is_valid:
                        intersection = buffered_geom1.intersection(buffered_geom2)
                        return GeometryProcessor.repair_geometry(intersection)
                except Exception:
                    pass
                    
            return None
            
        except Exception:
            return None
    
    def _geometry_to_coords(self, geom: BaseGeometry) -> List[List[float]]:
        """
        将几何对象转换为坐标列表
        
        Args:
            geom: 几何对象
            
        Returns:
            坐标列表 [[x, y], ...]
        """
        if not geom or geom.is_empty:
            return []
        
        geom_type = geom.geom_type
        
        if geom_type == 'LineString':
            return list(geom.coords)
        elif geom_type == 'MultiLineString':
            # 返回第一条线（或合并所有线）
            if geom.geoms:
                return list(geom.geoms[0].coords)
        elif geom_type == 'Polygon':
            return list(geom.exterior.coords)
        elif geom_type == 'GeometryCollection':
            # 尝试提取第一个有效几何
            for g in geom.geoms:
                coords = self._geometry_to_coords(g)
                if coords:
                    return coords
        
        return []

