"""
Metadata 构建器

生成 metadata.glb 文件（nuviz/metadata 消息）。
"""

from io import BytesIO
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image
from pyquaternion import Quaternion
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE, JOIN_STYLE

try:
    from .glb_encoder import GLBEncoder
    from .coord_utils import quat_to_wxyz
except ImportError:
    from glb_encoder import GLBEncoder
    from coord_utils import quat_to_wxyz

Image.MAX_IMAGE_PIXELS = 400000 * 400000


CAMERA_CHANNELS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# nuScenes 类别名称 -> ID 映射
NUSCENES_CATEGORIES = {
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
    "driveable_surface":    11,
    "other_flat":           12,
    "sidewalk":             13,
    "terrain":              14,
    "manmade":              15,
    "vegetation":           16,
}

# 各地图的画布边界 [xmin, ymin, xmax, ymax]（单位：米）
_MAP_BOUNDS: Dict[str, List[float]] = {
    "singapore-onenorth":       [-300.0,  -1500.0, 1500.0, 1000.0],
    "singapore-hollandvillage": [-300.0,  -1500.0, 1500.0, 1000.0],
    "singapore-queenstown":     [-300.0,  -1500.0, 1500.0, 1000.0],
    "boston-seaport":           [-2000.0, -1000.0, 2000.0, 2000.0],
}
_DEFAULT_MAP_BOUNDS: List[float] = [-2000.0, -2000.0, 2000.0, 2000.0]

# 需要提取的矢量地图图层
MAP_LAYERS = [
    'drivable_area',
    'road_segment',
    'lane',
    'lane_connector',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
]

# 轨迹缓冲半径（米），覆盖自车两侧约 21 条车道宽度
BUFFER_RADIUS_M = 75.0


class MetadataBuilder:
    """从 NuScenes 场景构建 nuviz/metadata GLB 文件。"""

    def __init__(
        self,
        nusc: NuScenes,
        scene_token: str,
        dataroot: Optional[str] = None,
        sample_tokens: Optional[List[str]] = None,
    ):
        self.nusc = nusc
        self.scene = nusc.get('scene', scene_token)
        # dataroot 用于实例化 NuScenesMap；若未传入则从 nusc 对象获取
        self.dataroot = dataroot or nusc.dataroot
        self.sample_tokens = sample_tokens

    def build(self) -> bytes:
        """构建并返回 metadata.glb 的字节内容。"""
        scene = self.scene
        log_rec = self.nusc.get('log', scene['log_token'])
        location = log_rec['location']

        sample_tokens = self._get_sample_tokens()
        first_sample = self.nusc.get('sample', sample_tokens[0])
        last_sample  = self.nusc.get('sample', sample_tokens[-1])
        start_time = first_sample['timestamp'] / 1e6
        end_time   = last_sample['timestamp']  / 1e6

        encoder = GLBEncoder()

        cameras = self._build_cameras(sample_tokens[0])

        map_json = self._build_map(encoder)
        statistics = self._build_statistics(encoder, sample_tokens, start_time)

        streams = {
            "/ego_pose": {"category": "POSE", "type": "pose", "coordinate": "world"},
            "/lidar": {"category": "PRIMITIVE", "type": "point", "coordinate": "world"},
            "/map/basemap": {"category": "PRIMITIVE", "type": "image", "coordinate": "world"},
            "/gt/objects/bounds": {"category": "PRIMITIVE", "type": "cuboid", "coordinate": "world"},
            "/gt/objects/future_trajectories": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
            "/gt/ego/future_trajectory": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
            "/pred/sparsedrive/planning": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
            "/pred/sparsedrive/objects/bounds": {
                "category": "PRIMITIVE",
                "type": "cuboid",
                "coordinate": "world",
            },
            "/pred/sparsedrive/map/divider": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
            "/pred/sparsedrive/map/boundary": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
            "/pred/sparsedrive/map/ped_crossing": {
                "category": "PRIMITIVE",
                "type": "polyline",
                "coordinate": "world",
            },
        }
        for channel in cameras:
            streams[f"/camera/{channel}"] = {
                "category": "PRIMITIVE",
                "type": "image",
                "coordinate": "ego",
            }
        for layer_name in MAP_LAYERS:
            streams[f"/gt/map/{layer_name}"] = {
                "category": "PRIMITIVE",
                "type": "polygon",
                "coordinate": "world",
            }

        metadata = {
            "type": "nuviz/metadata",
            "data": {
                "log_info": {
                    "start_time": start_time,
                    "end_time":   end_time,
                },
                "streams":    streams,
                "cameras":    cameras,
                "map":        map_json,
                "statistics": statistics,
                "extensions": {
                    "nuscenes": {
                        "scene": {
                            "scene_token": scene['token'],
                            "name":        scene['name'],
                            "description": scene['description'],
                            "location":    location,
                            "mapId":       location,
                        },
                        "map": {
                            "canvas_edge_m": _MAP_BOUNDS.get(location, _DEFAULT_MAP_BOUNDS),
                        },
                        "coordinate": {
                            "units":            "meter",
                            "matrixConvention": "nuscenes",
                            "quatOrder":        "wxyz",
                        },
                        "mapping": {
                            "classes": {
                                "nameToId": NUSCENES_CATEGORIES,
                            }
                        },
                    }
                },
            },
        }

        return encoder.encode(metadata)

    def _get_sample_tokens(self) -> List[str]:
        if self.sample_tokens:
            return self.sample_tokens

        tokens: List[str] = []
        sample_token = self.scene['first_sample_token']
        while sample_token:
            tokens.append(sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        if not tokens:
            raise ValueError(f"Scene has no samples: {self.scene['token']}")
        return tokens

    def _build_cameras(self, sample_token: str) -> Dict[str, Any]:
        """读取第一帧各相机的内外参，返回相机信息字典。"""
        sample = self.nusc.get('sample', sample_token)
        cameras = {}

        for channel in CAMERA_CHANNELS:
            if channel not in sample['data']:
                continue

            cam_data = self.nusc.get('sample_data', sample['data'][channel])
            cs_rec   = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

            cameras[channel] = {
                "image_width":  cam_data['width'],
                "image_height": cam_data['height'],
                "intrinsic":    np.array(cs_rec['camera_intrinsic']).tolist(),
                "extrinsic": {
                    "translation": cs_rec['translation'],
                    "rotation":    quat_to_wxyz(Quaternion(cs_rec['rotation'])),
                },
            }

        return cameras

    def _collect_trajectory(self) -> List[List[float]]:
        """
        按时间顺序收集场景内所有帧的 ego_pose XY 坐标。

        Returns:
            trajectory_xy: list of [x, y]，世界坐标系，单位米
        """
        sample_tokens = self._get_sample_tokens()
        trajectory_xy = []

        for sample_token in sample_tokens:
            sample   = self.nusc.get('sample', sample_token)
            lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ep       = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            trajectory_xy.append(ep['translation'][:2])

        return trajectory_xy

    def _build_statistics(
        self,
        encoder: GLBEncoder,
        sample_tokens: List[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """构建与 message_index.messages 对齐的帧统计。"""
        timestamps: List[float] = []
        positions: List[List[float]] = []

        for sample_token in sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ep = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            timestamps.append(sample['timestamp'] / 1e6)
            positions.append(list(ep['translation']))

        timeline = np.array([ts - start_time for ts in timestamps], dtype=np.float32)
        positions_arr = np.array(positions, dtype=np.float32)
        speed = np.zeros(len(sample_tokens), dtype=np.float32)
        acceleration = np.zeros(len(sample_tokens), dtype=np.float32)

        if len(sample_tokens) > 1:
            dt = np.diff(np.array(timestamps, dtype=np.float64))
            distances = np.linalg.norm(np.diff(positions_arr[:, :2], axis=0), axis=1)
            interval_speed = np.divide(
                distances,
                dt,
                out=np.zeros_like(distances, dtype=np.float64),
                where=dt > 0,
            )
            speed[0] = interval_speed[0]
            speed[1:] = interval_speed.astype(np.float32)
            interval_accel = np.diff(speed) / np.maximum(dt, 1e-6)
            acceleration[1:] = interval_accel.astype(np.float32)

        timeline_acc = encoder.add_accessor(timeline, type_str="SCALAR")
        speed_acc = encoder.add_accessor(speed, type_str="SCALAR")
        acceleration_acc = encoder.add_accessor(acceleration, type_str="SCALAR")

        statistics: Dict[str, Any] = {
            "frame_count": len(sample_tokens),
            "timeline": {
                "values": f"#/accessors/{timeline_acc}",
                "unit": "second",
                "reference": "log_info.start_time",
            },
            "ego_state": {
                "speed": {
                    "values": f"#/accessors/{speed_acc}",
                    "unit": "meter_per_second",
                },
                "acceleration": {
                    "values": f"#/accessors/{acceleration_acc}",
                    "unit": "meter_per_second_squared",
                },
            },
            "object_counts": {},
        }

        gt_counts = self._build_gt_object_count_statistics(encoder, sample_tokens)
        if gt_counts:
            statistics["object_counts"]["/gt/objects/bounds"] = gt_counts
        return statistics

    def _build_gt_object_count_statistics(
        self,
        encoder: GLBEncoder,
        sample_tokens: List[str],
    ) -> Dict[str, Any]:
        total_frames: List[int] = []
        total_values: List[int] = []
        category_frames: Dict[str, List[int]] = defaultdict(list)
        category_values: Dict[str, List[int]] = defaultdict(list)

        for frame_idx, sample_token in enumerate(sample_tokens):
            sample = self.nusc.get('sample', sample_token)
            if sample['anns']:
                total_frames.append(frame_idx)
                total_values.append(len(sample['anns']))

            counts: Dict[str, int] = defaultdict(int)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                category = self._resolve_gt_category(ann['category_name'])
                counts[category] += 1

            for category, count in counts.items():
                category_frames[category].append(frame_idx)
                category_values[category].append(count)

        if not total_frames:
            return {}

        total_frame_acc = encoder.add_accessor(np.array(total_frames, dtype=np.uint32), type_str="SCALAR")
        total_value_acc = encoder.add_accessor(np.array(total_values, dtype=np.uint32), type_str="SCALAR")

        payload: Dict[str, Any] = {
            "unit": "count",
            "total": {
                "frame_indices": f"#/accessors/{total_frame_acc}",
                "values": f"#/accessors/{total_value_acc}",
            },
            "categories": {},
        }

        for category in sorted(category_frames):
            frame_acc = encoder.add_accessor(np.array(category_frames[category], dtype=np.uint32), type_str="SCALAR")
            value_acc = encoder.add_accessor(np.array(category_values[category], dtype=np.uint32), type_str="SCALAR")
            payload["categories"][category] = {
                "frame_indices": f"#/accessors/{frame_acc}",
                "values": f"#/accessors/{value_acc}",
            }
        return payload

    def _resolve_gt_category(self, category_name: str) -> str:
        if "vehicle.car" in category_name:
            return "car"
        if "vehicle.truck" in category_name:
            return "truck"
        if "vehicle.bus" in category_name:
            return "bus"
        if "vehicle.trailer" in category_name:
            return "trailer"
        if "vehicle.construction" in category_name:
            return "construction_vehicle"
        if "vehicle.motorcycle" in category_name:
            return "motorcycle"
        if "vehicle.bicycle" in category_name:
            return "bicycle"
        if "pedestrian" in category_name:
            return "pedestrian"
        if "barrier" in category_name:
            return "barrier"
        if "traffic_cone" in category_name:
            return "traffic_cone"
        return category_name.split('.')[0]

    def _build_map(self, encoder: GLBEncoder) -> Dict[str, Any]:
        """
        提取场景矢量地图，将几何数据写入 GLBEncoder，
        返回写入 nuviz.data.map 的 JSON 字典。

        Args:
            encoder: GLBEncoder 实例，地图顶点/计数数据将追加至其 BIN chunk

        Returns:
            map_json: {"/gt/map/<layer>": {"vertices": ..., "offsets": ..., "count": ...}, ...}
        """
        # ① 收集轨迹
        trajectory_xy = self._collect_trajectory()

        # ② 构建缓冲多边形（Shapely 1.x API）
        if len(trajectory_xy) >= 2:
            line_or_point = LineString(trajectory_xy)
            buffer_poly = line_or_point.buffer(
                BUFFER_RADIUS_M,
                cap_style=CAP_STYLE.round,
                join_style=JOIN_STYLE.round,
            )
        else:
            from shapely.geometry import Point

            line_or_point = Point(trajectory_xy[0])
            buffer_poly = line_or_point.buffer(BUFFER_RADIUS_M)
        # patch_box: (xmin, ymin, xmax, ymax)，用于 API 粗筛
        xmin, ymin, xmax, ymax = buffer_poly.bounds
        patch_box = (xmin, ymin, xmax, ymax)

        # ③ 实例化 NuScenesMap
        scene   = self.scene
        log_rec = self.nusc.get('log', scene['log_token'])
        location = log_rec['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=location)

        # ④ 逐图层粗筛 + 精筛 + 顶点提取
        map_layers_json: Dict[str, Any] = {}

        for layer_name in MAP_LAYERS:
            # 粗筛：矩形查询
            records_in_patch = nusc_map.get_records_in_patch(
                patch_box, [layer_name], mode='intersect'
            )
            candidate_tokens = records_in_patch.get(layer_name, [])

            all_vertices: List[List[float]] = []
            all_counts:   List[int]         = []

            for token in candidate_tokens:
                record = nusc_map.get(layer_name, token)

                # 按图层类型选取正确的几何来源：
                # - drivable_area / lane_connector 使用 polygon_tokens（复数），
                #   每个元素才是真实轮廓多边形；其单数字段 polygon_token 仅是
                #   空间索引粗筛用的 AABB 包围盒，渲染时禁止使用，否则
                #   lane_connector 会全部退化为轴对齐黄色矩形。
                # - 当 lane_connector 无复数字段时，退回到 arcline_path_3
                #   离散化中心线并向两侧 buffer 1.75m 得到近似轮廓。
                # - 其余图层（lane、road_segment 等）用 polygon_token（单数）。
                geoms = []
                if 'polygon_tokens' in record and record['polygon_tokens']:
                    for pt in record['polygon_tokens']:
                        try:
                            geoms.append(nusc_map.extract_polygon(pt))
                        except Exception:
                            continue
                elif layer_name == 'lane_connector':
                    from nuscenes.map_expansion.arcline_path_utils import discretize_lane
                    from shapely.geometry import LineString as _LS
                    arc_path = nusc_map.arcline_path_3.get(token, [])
                    if arc_path:
                        try:
                            poses = discretize_lane(arc_path, resolution_meters=0.5)
                            if len(poses) >= 2:
                                cl = _LS([(p[0], p[1]) for p in poses])
                                geoms.append(cl.buffer(1.75, cap_style=2, join_style=2))
                        except Exception:
                            continue
                elif 'polygon_token' in record and record['polygon_token'] is not None:
                    try:
                        geoms.append(nusc_map.extract_polygon(record['polygon_token']))
                    except Exception:
                        continue

                for geom in geoms:
                    # 精筛：与缓冲多边形几何相交
                    if not buffer_poly.intersects(geom):
                        continue

                    # 裁剪：求交后只保留缓冲区内的部分。
                    # 对于 drivable_area 等存在覆盖整张地图的超大多边形的图层，
                    # 仅用 intersects() 精筛无法限制范围；必须通过 intersection()
                    # 将几何裁剪到 buffer_poly 边界内，否则整张地图的路网会被
                    # 全量写入 accessor，导致可视化时地图范围远超自车轨迹区域。
                    try:
                        clipped = buffer_poly.intersection(geom)
                    except Exception:
                        continue

                    # intersection 结果可能是 Polygon / MultiPolygon / GeometryCollection
                    from shapely.geometry import MultiPolygon, GeometryCollection
                    if clipped.is_empty:
                        continue
                    if hasattr(clipped, 'exterior'):
                        clip_polys = [clipped]
                    elif isinstance(clipped, (MultiPolygon, GeometryCollection)):
                        clip_polys = [g for g in clipped.geoms if hasattr(g, 'exterior')]
                    else:
                        continue

                    for cp in clip_polys:
                        # 提取外轮廓顶点，去掉闭合重复点
                        coords = list(cp.exterior.coords)[:-1]
                        if len(coords) < 3:
                            continue
                        for x, y in coords:
                            all_vertices.append([x, y, 0.0])
                        all_counts.append(len(coords))

            # 空图层不写入
            if not all_vertices:
                continue

            v_arr = np.array(all_vertices, dtype=np.float32)  # (K, 3)
            offsets = [0]
            for count in all_counts:
                offsets.append(offsets[-1] + count)
            o_arr = np.array(offsets, dtype=np.uint32)

            v_acc = encoder.add_accessor(v_arr, type_str="VEC3")
            o_acc = encoder.add_accessor(o_arr, type_str="SCALAR")

            map_layers_json[f"/gt/map/{layer_name}"] = {
                "vertices": f"#/accessors/{v_acc}",
                "offsets": f"#/accessors/{o_acc}",
                "count": len(all_counts),
            }

        map_json = map_layers_json

        basemap_json = self._build_basemap(encoder, nusc_map, location, buffer_poly.bounds)
        if basemap_json is not None:
            map_json["/map/basemap"] = basemap_json

        return map_json

    def _build_basemap(
        self,
        encoder: GLBEncoder,
        nusc_map: NuScenesMap,
        location: str,
        world_bounds: Tuple[float, float, float, float],
    ) -> Optional[Dict[str, Any]]:
        """
        从 nuScenes 原始 basemap.png 裁剪场景栅格底图，并写入 GLB images。

        world_bounds 为世界坐标系下的 (min_x, min_y, max_x, max_y)，与
        矢量地图使用同一个轨迹 buffer 范围，保证栅格图和矢量图对齐。
        """
        basemap_path = Path(self.dataroot) / "maps" / "basemap" / f"{location}.png"
        if not basemap_path.exists():
            return None

        source = Image.open(basemap_path).convert("RGB")
        pixel_box, actual_bounds = self._world_bounds_to_pixel_box(
            world_bounds,
            source.size,
            nusc_map.canvas_edge,
        )
        cropped = source.crop(pixel_box)

        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        image_idx = encoder.add_image(
            buffer.getvalue(),
            "image/png",
            cropped.width,
            cropped.height,
        )

        min_x, min_y, max_x, max_y = actual_bounds
        meters_per_pixel_x = (max_x - min_x) / cropped.width
        meters_per_pixel_y = (max_y - min_y) / cropped.height

        return {
            "image": f"#/images/{image_idx}",
            "mimeType": "image/png",
            "width": cropped.width,
            "height": cropped.height,
            "bounds": {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            },
            "resolution": {
                "meters_per_pixel_x": meters_per_pixel_x,
                "meters_per_pixel_y": meters_per_pixel_y,
            },
            "transform": {
                "origin_x": min_x,
                "origin_y": max_y,
                "y_axis_direction": "down",
            },
            "source": {
                "mapId": location,
                "layer": "basemap",
                "path": f"maps/basemap/{location}.png",
                "canvas_edge_m": [
                    0.0,
                    0.0,
                    float(nusc_map.canvas_edge[0]),
                    float(nusc_map.canvas_edge[1]),
                ],
                "pixel_box": list(pixel_box),
            },
        }

    def _world_bounds_to_pixel_box(
        self,
        bounds: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        canvas_edge: Tuple[float, float],
    ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        """
        将地图世界坐标范围转换为 basemap.png 的像素裁剪框。

        nuScenes basemap 图片覆盖 [0, canvas_width] x [0, canvas_height]，
        图片坐标原点在左上角，世界坐标 y 轴向上，因此 y 方向需要翻转。
        """
        min_x, min_y, max_x, max_y = bounds
        image_w, image_h = image_size
        canvas_w, canvas_h = float(canvas_edge[0]), float(canvas_edge[1])

        left = int(np.floor(min_x / canvas_w * image_w))
        right = int(np.ceil(max_x / canvas_w * image_w))
        top = int(np.floor((canvas_h - max_y) / canvas_h * image_h))
        bottom = int(np.ceil((canvas_h - min_y) / canvas_h * image_h))

        left = max(0, min(image_w, left))
        right = max(0, min(image_w, right))
        top = max(0, min(image_h, top))
        bottom = max(0, min(image_h, bottom))

        if left >= right or top >= bottom:
            raise ValueError(
                "Basemap crop is empty after clamping: "
                f"bounds={bounds}, pixel_box={(left, top, right, bottom)}"
            )

        actual_min_x = left / image_w * canvas_w
        actual_max_x = right / image_w * canvas_w
        actual_max_y = canvas_h - top / image_h * canvas_h
        actual_min_y = canvas_h - bottom / image_h * canvas_h

        return (
            left,
            top,
            right,
            bottom,
        ), (
            float(actual_min_x),
            float(actual_min_y),
            float(actual_max_x),
            float(actual_max_y),
        )
