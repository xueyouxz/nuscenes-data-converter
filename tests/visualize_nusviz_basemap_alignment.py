"""
Visualize NUSVIZ vector map over a cropped nuScenes raster basemap.

This is a verification script for checking whether a scene-level basemap crop
from the original nuScenes `maps/basemap/<mapId>.png` aligns with the vector map
stored in NUSVIZ `metadata.glb`.

Usage:
    python tests/visualize_nusviz_basemap_alignment.py \
        --scene-dir nusviz/output/scene-0916 \
        --dataroot /home/public/nuscenes_datasets/nuscenes-trainval

The output image is saved to:
    <scene-dir>/basemap_alignment_<scene-name>.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nusviz-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

Image.MAX_IMAGE_PIXELS = 400000 * 400000


REPO_ROOT = Path(__file__).resolve().parents[1]
NUSVIZ_ROOT = REPO_ROOT / "nusviz"
for path in (REPO_ROOT, NUSVIZ_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


LAYER_STYLE = {
    "drivable_area": dict(fc="#65B66E", ec="#347A3B", lw=0.5, alpha=0.22, zorder=2),
    "road_segment": dict(fc="#D8D8D8", ec="#555555", lw=0.5, alpha=0.24, zorder=3),
    "lane": dict(fc="#F3D26A", ec="#9F7A18", lw=0.45, alpha=0.24, zorder=4),
    "lane_connector": dict(fc="#E9A83A", ec="#8F5F0D", lw=0.45, alpha=0.24, zorder=4),
    "ped_crossing": dict(fc="#F36C6C", ec="#AE2E2E", lw=0.7, alpha=0.45, zorder=6),
    "walkway": dict(fc="#7ED4D0", ec="#2D817C", lw=0.5, alpha=0.30, zorder=5),
    "stop_line": dict(fc="#FF2D55", ec="#A60022", lw=1.0, alpha=0.70, zorder=7),
    "carpark_area": dict(fc="#B48AF0", ec="#6843A5", lw=0.5, alpha=0.24, zorder=4),
}


def parse_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"Not a glTF 2.0 GLB file: {path}")

    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk in GLB: {path}")

    json_start = 20
    json_data = json.loads(data[json_start : json_start + json_len].decode("utf-8"))

    bin_header = json_start + json_len
    if bin_header + 8 > len(data):
        return json_data, b""

    bin_len, bin_type = struct.unpack_from("<II", data, bin_header)
    if bin_type != 0x004E4942:
        return json_data, b""

    bin_start = bin_header + 8
    return json_data, data[bin_start : bin_start + bin_len]


def read_accessor(json_data: Dict[str, Any], bin_data: bytes, accessor_ref: str) -> np.ndarray:
    accessor_idx = int(accessor_ref.split("/")[-1])
    accessor = json_data["accessors"][accessor_idx]
    buffer_view = json_data["bufferViews"][accessor["bufferView"]]

    dtype_map = {
        5126: np.float32,
        5125: np.uint32,
        5123: np.uint16,
        5121: np.uint8,
        5122: np.int16,
        5120: np.int8,
    }
    cols_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    dtype = dtype_map[accessor["componentType"]]
    cols = cols_map[accessor["type"]]
    count = accessor["count"]
    byte_offset = buffer_view["byteOffset"]
    byte_length = buffer_view["byteLength"]
    raw = bin_data[byte_offset : byte_offset + byte_length]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(count, cols) if cols > 1 else arr


def extract_layer_polygons(
    json_data: Dict[str, Any],
    bin_data: bytes,
    layer: Dict[str, str],
) -> List[np.ndarray]:
    vertices = read_accessor(json_data, bin_data, layer["vertices"])
    if "offsets" in layer:
        offsets = read_accessor(json_data, bin_data, layer["offsets"])
    else:
        counts = read_accessor(json_data, bin_data, layer["counts"])
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)

    polygons = []
    for idx in range(len(offsets) - 1):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        poly = vertices[start:end, :2]
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def iter_vector_polygons(json_data: Dict[str, Any], bin_data: bytes) -> Iterable[Tuple[str, np.ndarray]]:
    map_data = json_data["nuviz"]["data"].get("map", {})
    layers = map_data.get("layers")
    if not layers:
        layers = {
            stream_name.removeprefix("/gt/map/"): payload
            for stream_name, payload in map_data.items()
            if stream_name.startswith("/gt/map/") and "vertices" in payload
        }
    for layer_name, layer in layers.items():
        for polygon in extract_layer_polygons(json_data, bin_data, layer):
            yield layer_name, polygon


def collect_ego_xy(scene_dir: Path) -> Optional[np.ndarray]:
    index_path = scene_dir / "message_index.json"
    if not index_path.exists():
        return None

    index = json.loads(index_path.read_text())
    ego_xy = []
    for entry in index.get("messages", []):
        frame_path = scene_dir / entry["file"]
        if not frame_path.exists():
            continue
        frame_json, _ = parse_glb(frame_path)
        updates = frame_json.get("nuviz", {}).get("data", {}).get("updates", [])
        if not updates:
            continue
        ego_pose = updates[0].get("poses", {}).get("/ego_pose")
        if ego_pose is None:
            continue
        translation = ego_pose["translation"]
        ego_xy.append([translation[0], translation[1]])

    return np.asarray(ego_xy, dtype=np.float64) if ego_xy else None


def compute_crop_bounds(
    ego_xy: Optional[np.ndarray],
    vector_polygons: List[np.ndarray],
    buffer_radius_m: float,
) -> Tuple[float, float, float, float]:
    if ego_xy is not None and len(ego_xy) >= 2:
        try:
            from shapely.geometry import LineString
            from shapely.geometry import CAP_STYLE, JOIN_STYLE

            line = LineString(ego_xy[:, :2])
            buffer_poly = line.buffer(
                buffer_radius_m,
                cap_style=CAP_STYLE.round,
                join_style=JOIN_STYLE.round,
            )
            return tuple(float(v) for v in buffer_poly.bounds)
        except Exception as exc:
            print(f"WARN: failed to compute trajectory buffer with shapely: {exc}")

    if vector_polygons:
        xy = np.concatenate(vector_polygons, axis=0)
        min_x, min_y = xy.min(axis=0)
        max_x, max_y = xy.max(axis=0)
        margin = 5.0
        return (
            float(min_x - margin),
            float(min_y - margin),
            float(max_x + margin),
            float(max_y + margin),
        )

    raise ValueError("Cannot compute crop bounds: no ego trajectory or vector polygons found")


def load_map_canvas(dataroot: Path, location: str) -> Tuple[float, float]:
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap

        nusc_map = NuScenesMap(dataroot=str(dataroot), map_name=location)
        canvas_w, canvas_h = nusc_map.canvas_edge
        return float(canvas_w), float(canvas_h)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load NuScenesMap canvas_edge. Check --dataroot and map expansion files."
        ) from exc


def world_bounds_to_pixel_box(
    bounds: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    canvas_size_m: Tuple[float, float],
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = bounds
    image_w, image_h = image_size
    canvas_w, canvas_h = canvas_size_m

    left = math.floor(min_x / canvas_w * image_w)
    right = math.ceil(max_x / canvas_w * image_w)
    top = math.floor((canvas_h - max_y) / canvas_h * image_h)
    bottom = math.ceil((canvas_h - min_y) / canvas_h * image_h)

    left = max(0, min(image_w, left))
    right = max(0, min(image_w, right))
    top = max(0, min(image_h, top))
    bottom = max(0, min(image_h, bottom))

    if left >= right or top >= bottom:
        raise ValueError(
            f"Empty crop after clamping. world_bounds={bounds}, pixel_box={(left, top, right, bottom)}"
        )

    actual_min_x = left / image_w * canvas_w
    actual_max_x = right / image_w * canvas_w
    actual_max_y = canvas_h - top / image_h * canvas_h
    actual_min_y = canvas_h - bottom / image_h * canvas_h

    return (left, top, right, bottom), (
        float(actual_min_x),
        float(actual_min_y),
        float(actual_max_x),
        float(actual_max_y),
    )


def crop_basemap(
    dataroot: Path,
    location: str,
    crop_bounds: Tuple[float, float, float, float],
) -> Tuple[Image.Image, Tuple[float, float, float, float], Tuple[int, int, int, int]]:
    basemap_path = dataroot / "maps" / "basemap" / f"{location}.png"
    if not basemap_path.exists():
        raise FileNotFoundError(f"Basemap image not found: {basemap_path}")

    source = Image.open(basemap_path).convert("RGB")
    canvas_size = load_map_canvas(dataroot, location)
    pixel_box, actual_bounds = world_bounds_to_pixel_box(crop_bounds, source.size, canvas_size)
    return source.crop(pixel_box), actual_bounds, pixel_box


def plot_alignment(
    scene_dir: Path,
    dataroot: Path,
    output_path: Optional[Path],
    padding_m: float,
) -> Path:
    metadata_path = scene_dir / "metadata.glb"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.glb not found: {metadata_path}")

    json_data, bin_data = parse_glb(metadata_path)
    nuviz_data = json_data["nuviz"]["data"]
    scene_meta = nuviz_data["extensions"]["nuscenes"]["scene"]
    location = scene_meta["location"]
    scene_name = scene_meta["name"]
    map_data = nuviz_data.get("map", {})
    buffer_radius_m = float(map_data.get("buffer_radius_m", 75.0))

    layer_polygons = list(iter_vector_polygons(json_data, bin_data))
    vector_polygons = [polygon for _, polygon in layer_polygons]
    ego_xy = collect_ego_xy(scene_dir)

    min_x, min_y, max_x, max_y = compute_crop_bounds(ego_xy, vector_polygons, buffer_radius_m)
    crop_bounds = (
        min_x - padding_m,
        min_y - padding_m,
        max_x + padding_m,
        max_y + padding_m,
    )
    basemap_crop, actual_bounds, pixel_box = crop_basemap(dataroot, location, crop_bounds)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    ax.imshow(
        np.asarray(basemap_crop),
        extent=[actual_bounds[0], actual_bounds[2], actual_bounds[1], actual_bounds[3]],
        origin="upper",
        zorder=1,
    )

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#111111", label="Cropped basemap")
    ]

    layer_counts: Dict[str, int] = {}
    for layer_name, polygon in layer_polygons:
        style = LAYER_STYLE.get(
            layer_name,
            dict(fc="#CCCCCC", ec="#333333", lw=0.5, alpha=0.25, zorder=4),
        )
        patch = MplPolygon(polygon, closed=True)
        collection = PatchCollection(
            [patch],
            facecolor=style["fc"],
            edgecolor=style["ec"],
            linewidth=style["lw"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        ax.add_collection(collection)
        layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1

    for layer_name, count in sorted(layer_counts.items()):
        style = LAYER_STYLE.get(layer_name, dict(fc="#CCCCCC", ec="#333333", alpha=0.35))
        legend_handles.append(
            mpatches.Patch(
                facecolor=style["fc"],
                edgecolor=style["ec"],
                alpha=max(style.get("alpha", 0.35), 0.35),
                label=f"{layer_name} ({count})",
            )
        )

    if ego_xy is not None and len(ego_xy) > 0:
        ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="#FF2D55", linewidth=2.0, zorder=20)
        ax.scatter(ego_xy[0, 0], ego_xy[0, 1], color="#00A86B", s=60, zorder=21)
        ax.scatter(ego_xy[-1, 0], ego_xy[-1, 1], color="#1F5EFF", s=60, marker="s", zorder=21)
        legend_handles.extend(
            [
                mpatches.Patch(facecolor="#FF2D55", label="Ego trajectory"),
                mpatches.Patch(facecolor="#00A86B", label="Start"),
                mpatches.Patch(facecolor="#1F5EFF", label="End"),
            ]
        )

    ax.set_xlim(actual_bounds[0], actual_bounds[2])
    ax.set_ylim(actual_bounds[1], actual_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#222222", linewidth=0.4, linestyle="--", alpha=0.35)
    ax.set_xlabel("X / East (m)")
    ax.set_ylabel("Y / North (m)")
    ax.set_title(
        f"Basemap crop alignment - {scene_name} | {location} | "
        f"buffer={buffer_radius_m:g}m | px={pixel_box}"
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.88)

    fig.tight_layout()

    if output_path is None:
        output_path = scene_dir / f"basemap_alignment_{scene_dir.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"scene: {scene_name}")
    print(f"location: {location}")
    print(f"requested_bounds_m: {crop_bounds}")
    print(f"actual_bounds_m: {actual_bounds}")
    print(f"pixel_box: {pixel_box}")
    print(f"saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay NUSVIZ vector map on a scene crop from nuScenes basemap.png."
    )
    parser.add_argument("--scene-dir", required=True, type=Path, help="NUSVIZ scene directory.")
    parser.add_argument("--dataroot", required=True, type=Path, help="nuScenes dataroot.")
    parser.add_argument("--output", default=None, type=Path, help="Output PNG path.")
    parser.add_argument(
        "--padding-m",
        default=0.0,
        type=float,
        help="Extra padding around the trajectory-buffer crop bounds in meters.",
    )
    args = parser.parse_args()

    plot_alignment(
        scene_dir=args.scene_dir,
        dataroot=args.dataroot,
        output_path=args.output,
        padding_m=args.padding_m,
    )


if __name__ == "__main__":
    main()
