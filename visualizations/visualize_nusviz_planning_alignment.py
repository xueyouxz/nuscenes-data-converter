#!/usr/bin/env python3
"""
Visualize NUSVIZ GT ego trajectory and SparseDrive planning trajectories.

The script reads an already converted NUSVIZ scene directory and overlays:
- the real ego trajectory from /ego_pose in every keyframe message
- the GT future trajectory stream /ego/fut_trajectory when present
- the SparseDrive final_planning stream /ego/planning_trajectory for sampled keyframes

Example:
    python visualizations/visualize_nusviz_planning_alignment.py \
        --scene-dir nusviz/output/scene-0916 \
        --output planning_alignment_scene-0916.png \
        --stride 2
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PLANNING_STREAM = "/ego/planning_trajectory"
EGO_FUT_STREAM = "/ego/fut_trajectory"


def parse_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"Not a glTF 2.0 GLB file: {path}")

    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk: {path}")

    json_data = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_header_offset = 20 + json_len
    if bin_header_offset >= len(data):
        return json_data, b""

    bin_len, bin_type = struct.unpack_from("<II", data, bin_header_offset)
    if bin_type != 0x004E4942:
        raise ValueError(f"Missing BIN chunk: {path}")
    bin_data = data[bin_header_offset + 8 : bin_header_offset + 8 + bin_len]
    return json_data, bin_data


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
    offset = buffer_view["byteOffset"]
    length = buffer_view["byteLength"]

    raw = bin_data[offset : offset + length]
    arr = np.frombuffer(raw, dtype=dtype, count=count * cols)
    return arr.reshape(count, cols) if cols > 1 else arr


def extract_trajectory(
    json_data: Dict[str, Any],
    bin_data: bytes,
    update: Dict[str, Any],
    stream_name: str,
) -> Optional[np.ndarray]:
    primitive = update.get("primitives", {}).get(stream_name)
    if primitive is None:
        return None

    trajectory = primitive.get("trajectory", [])
    if not trajectory:
        return None

    poses = read_accessor(json_data, bin_data, trajectory[0]["poses"])
    return poses[:, :2]


def load_scene(scene_dir: Path) -> Dict[str, Any]:
    index_path = scene_dir / "message_index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    message_index = json.loads(index_path.read_text())
    frames: List[Dict[str, Any]] = []

    for entry in message_index["messages"]:
        frame_path = scene_dir / entry["file"]
        json_data, bin_data = parse_glb(frame_path)
        update = json_data["nuviz"]["data"]["updates"][0]

        ego_pose = np.asarray(update["poses"]["/ego_pose"]["translation"][:2], dtype=np.float32)
        planning = extract_trajectory(json_data, bin_data, update, PLANNING_STREAM)
        ego_future = extract_trajectory(json_data, bin_data, update, EGO_FUT_STREAM)

        start_error = None
        if planning is not None and len(planning) > 0:
            start_error = float(np.linalg.norm(planning[0] - ego_pose))

        frames.append(
            {
                "index": int(entry["index"]),
                "timestamp": float(entry["timestamp"]),
                "ego_pose": ego_pose,
                "planning": planning,
                "ego_future": ego_future,
                "planning_start_error_m": start_error,
            }
        )

    return {
        "message_index": message_index,
        "frames": frames,
        "ego_xy": np.asarray([frame["ego_pose"] for frame in frames], dtype=np.float32),
    }


def choose_frame_indices(frame_count: int, stride: int, max_frames: Optional[int]) -> List[int]:
    indices = list(range(0, frame_count, max(1, stride)))
    if max_frames is None or len(indices) <= max_frames:
        return indices

    positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
    return [indices[pos] for pos in positions]


def set_equal_view(ax: Any, xy_groups: List[np.ndarray], margin: float) -> None:
    points = []
    for group in xy_groups:
        if group is not None and len(group) > 0:
            points.append(group[:, :2])
    if not points:
        return

    merged = np.vstack(points)
    xmin, ymin = merged.min(axis=0)
    xmax, ymax = merged.max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_range = max(xmax - xmin, ymax - ymin) / 2 + margin
    ax.set_xlim(cx - half_range, cx + half_range)
    ax.set_ylim(cy - half_range, cy + half_range)


def visualize(scene_dir: Path, output: Optional[str], stride: int, max_frames: Optional[int], margin: float) -> None:
    scene = load_scene(scene_dir)
    frames = scene["frames"]
    ego_xy = scene["ego_xy"]
    selected_indices = choose_frame_indices(len(frames), stride, max_frames)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.plot(
        ego_xy[:, 0],
        ego_xy[:, 1],
        color="#111111",
        linewidth=2.4,
        marker="o",
        markersize=3,
        label="GT ego trajectory",
        zorder=5,
    )

    cmap = plt.get_cmap("viridis")
    plotted_planning = 0
    xy_groups: List[np.ndarray] = [ego_xy]
    start_errors: List[float] = []

    for order, frame_idx in enumerate(selected_indices):
        frame = frames[frame_idx]
        planning = frame["planning"]
        if planning is None or len(planning) == 0:
            continue

        color = cmap(order / max(1, len(selected_indices) - 1))
        label = "SparseDrive planning" if plotted_planning == 0 else None
        ax.plot(
            planning[:, 0],
            planning[:, 1],
            color=color,
            linewidth=1.6,
            alpha=0.8,
            label=label,
            zorder=4,
        )
        ax.scatter(
            planning[0, 0],
            planning[0, 1],
            color=color,
            s=20,
            edgecolors="white",
            linewidths=0.5,
            zorder=6,
        )

        if frame["planning_start_error_m"] is not None:
            start_errors.append(frame["planning_start_error_m"])

        xy_groups.append(planning)
        plotted_planning += 1

    if frames and frames[0]["ego_future"] is not None:
        ego_future = frames[0]["ego_future"]
        ax.plot(
            ego_future[:, 0],
            ego_future[:, 1],
            color="#E4572E",
            linestyle="--",
            linewidth=1.8,
            alpha=0.7,
            label="Frame 0 /ego/fut_trajectory",
            zorder=3,
        )
        xy_groups.append(ego_future)

    title = f"NUSVIZ planning alignment - {scene_dir.name}"
    if start_errors:
        title += f" | max start error {max(start_errors):.3f} m"
    ax.set_title(title)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    set_equal_view(ax, xy_groups, margin)
    ax.legend(loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(0, len(frames) - 1)))
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Keyframe index")

    summary = (
        f"frames={len(frames)}, sampled={len(selected_indices)}, "
        f"planning_drawn={plotted_planning}"
    )
    ax.text(
        0.01,
        0.01,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
    )

    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Saved visualization: {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay GT ego trajectory and NUSVIZ SparseDrive planning trajectories."
    )
    parser.add_argument("--scene-dir", required=True, help="NUSVIZ scene directory")
    parser.add_argument("--output", default=None, help="Optional output image path")
    parser.add_argument("--stride", type=int, default=1, help="Draw every Nth keyframe planning trajectory")
    parser.add_argument("--max-frames", type=int, default=80, help="Maximum sampled keyframes to draw")
    parser.add_argument("--margin", type=float, default=15.0, help="Plot margin in meters")
    args = parser.parse_args()

    visualize(
        scene_dir=Path(args.scene_dir),
        output=args.output,
        stride=args.stride,
        max_frames=args.max_frames,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
