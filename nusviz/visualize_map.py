"""
可视化 nusviz metadata.glb 中的矢量地图

用法：
    python visualize_map.py [scene_dir]  # 默认使用 output/scene-0103

输出：在屏幕显示地图，并保存为 map_<scene_name>.png
"""

import struct
import json
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无头模式，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon


# ─────────────────────────────────────────────
# 各图层的显示样式
# ─────────────────────────────────────────────

LAYER_STYLE = {
    'drivable_area':  dict(fc='#C8D8E8', ec='#7A9AB5', lw=0.6, alpha=0.8, zorder=1),
    'road_segment':   dict(fc='#D6D6D6', ec='#888888', lw=0.6, alpha=0.7, zorder=2),
    'lane':           dict(fc='#E8E0C8', ec='#B8A878', lw=0.5, alpha=0.6, zorder=3),
    'lane_connector': dict(fc='#F0D8A0', ec='#C8A840', lw=0.5, alpha=0.5, zorder=3),
    'ped_crossing':   dict(fc='#F0C8C8', ec='#D07070', lw=0.8, alpha=0.8, zorder=4),
    'walkway':        dict(fc='#C8E8C8', ec='#70A870', lw=0.6, alpha=0.7, zorder=4),
    'stop_line':      dict(fc='#F08080', ec='#C83030', lw=1.0, alpha=0.9, zorder=5),
    'carpark_area':   dict(fc='#E0C8F0', ec='#9060C0', lw=0.6, alpha=0.7, zorder=3),
}

LAYER_LABEL = {
    'drivable_area':  'Drivable Area',
    'road_segment':   'Road Segment',
    'lane':           'Lane',
    'lane_connector': 'Lane Connector',
    'ped_crossing':   'Ped Crossing',
    'walkway':        'Walkway',
    'stop_line':      'Stop Line',
    'carpark_area':   'Carpark Area',
}


# ─────────────────────────────────────────────
# GLB 解析工具
# ─────────────────────────────────────────────

def parse_glb(path):
    data = Path(path).read_bytes()
    magic, _, _ = struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67, f"Not a GLB file: {path}"

    json_len, _ = struct.unpack_from('<II', data, 12)
    json_data = json.loads(data[20 : 20 + json_len].decode('utf-8'))

    bin_offset = 20 + json_len
    bin_len, _ = struct.unpack_from('<II', data, bin_offset)
    bin_data = data[bin_offset + 8 : bin_offset + 8 + bin_len]

    return json_data, bin_data


def read_accessor(json_data, bin_data, accessor_ref):
    idx = int(accessor_ref.split('/')[-1])
    acc = json_data['accessors'][idx]
    bv  = json_data['bufferViews'][acc['bufferView']]
    dtype_map = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    cols_map  = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    dtype = dtype_map[acc['componentType']]
    cols  = cols_map[acc['type']]
    count = acc['count']
    raw = bin_data[bv['byteOffset'] : bv['byteOffset'] + bv['byteLength']]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(count, cols) if cols > 1 else arr


def extract_polygons(json_data, bin_data, layer):
    """从一个图层的 accessor 还原多边形列表，返回 list of (N,2) ndarray。"""
    vertices = read_accessor(json_data, bin_data, layer['vertices'])  # (K, 3)
    if 'offsets' in layer:
        offsets = read_accessor(json_data, bin_data, layer['offsets'])
    else:
        counts = read_accessor(json_data, bin_data, layer['counts'])
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)

    polygons = []
    for idx in range(len(offsets) - 1):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        polygons.append(vertices[start:end, :2])  # 只取 XY
    return polygons


def _map_layers(map_data):
    if map_data.get('layers'):
        return map_data['layers']
    return {
        stream_name.removeprefix('/gt/map/'): payload
        for stream_name, payload in map_data.items()
        if stream_name.startswith('/gt/map/') and 'vertices' in payload
    }


# ─────────────────────────────────────────────
# 主可视化函数
# ─────────────────────────────────────────────

def visualize_scene_map(scene_dir: str, save_path: str = None):
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    meta_path  = scene_dir / 'metadata.glb'

    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        sys.exit(1)

    json_data, bin_data = parse_glb(meta_path)
    nuviz    = json_data['nuviz']
    data     = nuviz['data']
    map_data = data.get('map')

    layers = _map_layers(map_data or {})
    if not layers:
        print("ERROR: No map data found in metadata.glb")
        sys.exit(1)

    location = data['extensions']['nuscenes']['scene']['location']
    radius   = map_data.get('buffer_radius_m', 75)

    # ── 读取自车轨迹（从 message_index.json + 各帧 ego_pose）──
    ego_xy = []
    idx_path = scene_dir / 'message_index.json'
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
        for entry in index['messages']:
            frame_path = scene_dir / entry['file']
            if frame_path.exists():
                fj, _ = parse_glb(frame_path)
                update = fj['nuviz']['data']['updates'][0]
                t = update['poses']['/ego_pose']['translation']
                ego_xy.append([t[0], t[1]])
    ego_xy = np.array(ego_xy) if ego_xy else None

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    legend_patches = []
    all_x, all_y = [], []

    for layer_name, layer in layers.items():
        style = LAYER_STYLE.get(layer_name, dict(fc='#CCCCCC', ec='#888888', lw=0.5, alpha=0.6, zorder=2))
        label = LAYER_LABEL.get(layer_name, layer_name)
        polygons = extract_polygons(json_data, bin_data, layer)

        patches = []
        for poly_xy in polygons:
            if len(poly_xy) < 3:
                continue
            patches.append(MplPolygon(poly_xy, closed=True))
            all_x.extend(poly_xy[:, 0].tolist())
            all_y.extend(poly_xy[:, 1].tolist())

        if patches:
            pc = PatchCollection(
                patches,
                facecolor=style['fc'],
                edgecolor=style['ec'],
                linewidth=style['lw'],
                alpha=style['alpha'],
                zorder=style['zorder'],
            )
            ax.add_collection(pc)
            legend_patches.append(
                mpatches.Patch(facecolor=style['fc'], edgecolor=style['ec'],
                               label=f"{label} ({len(polygons)})", alpha=0.9)
            )

    # 自车轨迹
    if ego_xy is not None and len(ego_xy) > 1:
        ax.plot(ego_xy[:, 0], ego_xy[:, 1],
                color='#FF6B35', linewidth=2.5, zorder=10, label='Ego trajectory')
        ax.scatter(ego_xy[0, 0],  ego_xy[0, 1],
                   c='#00FF88', s=80, zorder=11, marker='o', label='Start')
        ax.scatter(ego_xy[-1, 0], ego_xy[-1, 1],
                   c='#FF3366', s=80, zorder=11, marker='s', label='End')
        legend_patches += [
            mpatches.Patch(facecolor='#FF6B35', label='Ego trajectory'),
            mpatches.Patch(facecolor='#00FF88', label='Start'),
            mpatches.Patch(facecolor='#FF3366', label='End'),
        ]
        all_x.extend(ego_xy[:, 0].tolist())
        all_y.extend(ego_xy[:, 1].tolist())

    # 自动设置视口（加 10m 边距）
    if all_x and all_y:
        margin = 10
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    ax.set_xlabel('X / East (m)', color='#CCCCCC', fontsize=10)
    ax.set_ylabel('Y / North (m)', color='#CCCCCC', fontsize=10)
    ax.set_title(
        f'nusviz Vector Map  —  {scene_name}  |  {location}  |  buffer={radius}m',
        color='#EEEEEE', fontsize=13, pad=14,
    )
    ax.legend(
        handles=legend_patches, loc='upper right',
        fontsize=8, framealpha=0.3,
        facecolor='#2A2A4A', edgecolor='#555555',
        labelcolor='#DDDDDD',
    )
    ax.grid(True, color='#2E2E4E', linewidth=0.5, linestyle='--')

    plt.tight_layout()

    if save_path is None:
        save_path = str(scene_dir / f'map_{scene_name}.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    base_dir = Path(__file__).parent / 'output'

    if len(sys.argv) > 1:
        scene_dirs = [Path(sys.argv[1])]
    else:
        # 默认：渲染 output/ 下所有已生成的场景
        scene_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
        if not scene_dirs:
            print(f"No scene directories found in {base_dir}")
            sys.exit(1)

    for scene_dir in scene_dirs:
        meta = scene_dir / 'metadata.glb'
        if not meta.exists():
            print(f"SKIP {scene_dir.name}: no metadata.glb")
            continue
        print(f"Visualizing {scene_dir.name} ...")
        visualize_scene_map(str(scene_dir))

    print("Done.")
