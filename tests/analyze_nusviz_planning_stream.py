#!/usr/bin/env python3
"""
Analyze whether SparseDrive final_planning was written into the NUSVIZ stream.

This script does not generate NUSVIZ data. It validates an already converted
scene directory that contains metadata.glb, message_index.json, and messages/*.glb.

Basic structural check:
    python tests/analyze_nusviz_planning_stream.py \
        --scene-dir nusviz/output/scene-0916

Full source comparison against SparseDrive pkl:
    python tests/analyze_nusviz_planning_stream.py \
        --scene-dir nusviz/output/scene-0916 \
        --sparsedrive-pkl data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl \
        --nuscenes-dataroot /home/public/nuscenes_datasets/nuscenes-trainval \
        --version v1.0-trainval
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NUSVIZ_ROOT = REPO_ROOT / "nusviz"
for path in (REPO_ROOT, NUSVIZ_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PLANNING_STREAM = "/ego/planning_trajectory"
EGO_GT_STREAM = "/ego/fut_trajectory"


def parse_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise AssertionError(f"Not a glTF 2.0 GLB file: {path}")

    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise AssertionError(f"Missing JSON chunk: {path}")

    json_data = json.loads(data[20 : 20 + json_len].decode("utf-8"))

    bin_header_offset = 20 + json_len
    if bin_header_offset >= len(data):
        return json_data, b""

    bin_len, bin_type = struct.unpack_from("<II", data, bin_header_offset)
    if bin_type != 0x004E4942:
        raise AssertionError(f"Missing BIN chunk: {path}")
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


def get_update(scene_dir: Path, message_entry: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes, Dict[str, Any]]:
    json_data, bin_data = parse_glb(scene_dir / message_entry["file"])
    update = json_data["nuviz"]["data"]["updates"][0]
    return json_data, bin_data, update


def extract_stream_poses(
    json_data: Dict[str, Any],
    bin_data: bytes,
    update: Dict[str, Any],
    stream_name: str,
) -> Optional[np.ndarray]:
    primitive = update.get("primitives", {}).get(stream_name)
    if primitive is None:
        return None
    trajectory = primitive.get("trajectory", [])
    if len(trajectory) != 1:
        raise AssertionError(f"{stream_name} should contain exactly one trajectory")

    traj_meta = trajectory[0]
    poses = read_accessor(json_data, bin_data, traj_meta["poses"])
    if int(traj_meta["count"]) != len(poses):
        raise AssertionError(
            f"{stream_name} count mismatch: json={traj_meta['count']} accessor={len(poses)}"
        )
    return poses


def collect_sample_tokens(scene_token: str, dataroot: str, version: str) -> Tuple[Any, List[str]]:
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scene = nusc.get("scene", scene_token)
    sample_tokens: List[str] = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample_tokens.append(sample_token)
        sample_token = nusc.get("sample", sample_token)["next"]
    return nusc, sample_tokens


def build_sparsedrive_extractor(sparsedrive_pkl: str, nusc: Any) -> Any:
    from data_converter.core.sparsedrive_extractor import SparseDriveExtractor
    from nusviz.converter import _NuVizEgoPoseAdapter

    return SparseDriveExtractor(sparsedrive_pkl, _NuVizEgoPoseAdapter(nusc))


def validate_scene(args: argparse.Namespace) -> Dict[str, Any]:
    scene_dir = Path(args.scene_dir)
    metadata_path = scene_dir / "metadata.glb"
    index_path = scene_dir / "message_index.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    metadata_json, _ = parse_glb(metadata_path)
    streams = metadata_json["nuviz"]["data"].get("streams", {})
    if PLANNING_STREAM not in streams:
        raise AssertionError(f"metadata.glb does not declare {PLANNING_STREAM}")

    message_index = json.loads(index_path.read_text())
    messages = message_index["messages"]
    scene_token = message_index["extensions"]["nuscenes"]["scene_token"]

    sample_tokens: Optional[List[str]] = None
    sd_extractor = None
    if args.sparsedrive_pkl:
        if not args.nuscenes_dataroot:
            raise ValueError("--nuscenes-dataroot is required with --sparsedrive-pkl")
        nusc, sample_tokens = collect_sample_tokens(scene_token, args.nuscenes_dataroot, args.version)
        if len(sample_tokens) != len(messages):
            raise AssertionError(
                f"message/sample count mismatch: messages={len(messages)} samples={len(sample_tokens)}"
            )
        sd_extractor = build_sparsedrive_extractor(args.sparsedrive_pkl, nusc)

    frames_with_stream = 0
    frames_missing_stream = 0
    frames_compared_to_source = 0
    max_abs_xy_error = 0.0
    first_point_errors: List[float] = []
    per_frame: List[Dict[str, Any]] = []

    for idx, entry in enumerate(messages):
        json_data, bin_data, update = get_update(scene_dir, entry)
        planning = extract_stream_poses(json_data, bin_data, update, PLANNING_STREAM)
        ego_gt = extract_stream_poses(json_data, bin_data, update, EGO_GT_STREAM)
        ego_pose_xy = np.asarray(update["poses"]["/ego_pose"]["translation"][:2], dtype=np.float32)

        frame_report: Dict[str, Any] = {
            "frame_index": idx,
            "timestamp": entry["timestamp"],
            "has_planning_stream": planning is not None,
            "planning_point_count": 0 if planning is None else int(len(planning)),
        }

        if planning is None:
            frames_missing_stream += 1
        else:
            frames_with_stream += 1
            if planning.ndim != 2 or planning.shape[1] != 3:
                raise AssertionError(f"Frame {idx}: planning accessor shape should be (T, 3), got {planning.shape}")
            if len(planning) < 2:
                raise AssertionError(f"Frame {idx}: planning trajectory should include current point and future points")
            if not np.all(np.isfinite(planning)):
                raise AssertionError(f"Frame {idx}: planning trajectory contains NaN/Inf")
            if not np.allclose(planning[:, 2], 0.0, atol=args.atol):
                raise AssertionError(f"Frame {idx}: planning z values are not zero")

            first_point_error = float(np.linalg.norm(planning[0, :2] - ego_pose_xy))
            first_point_errors.append(first_point_error)
            frame_report["first_point_ego_error_m"] = first_point_error

            if ego_gt is not None and len(ego_gt) > 0:
                gt_start_error = float(np.linalg.norm(planning[0, :2] - ego_gt[0, :2]))
                frame_report["first_point_gt_error_m"] = gt_start_error

        if sd_extractor is not None and sample_tokens is not None:
            sample_token = sample_tokens[idx]
            has_source_prediction = sd_extractor.has_prediction(sample_token)
            frame_report["has_source_prediction"] = has_source_prediction

            if has_source_prediction:
                if planning is None:
                    raise AssertionError(f"Frame {idx}: source prediction exists but {PLANNING_STREAM} is missing")

                expected_xy = np.asarray(sd_extractor.extract_planning(sample_token), dtype=np.float32)
                if expected_xy.ndim != 2 or expected_xy.shape[1] != 2:
                    raise AssertionError(f"Frame {idx}: source planning shape should be (T, 2), got {expected_xy.shape}")
                if len(expected_xy) != len(planning):
                    raise AssertionError(
                        f"Frame {idx}: point count mismatch source={len(expected_xy)} stream={len(planning)}"
                    )

                abs_xy_error = np.abs(expected_xy - planning[:, :2])
                frame_max_error = float(abs_xy_error.max()) if abs_xy_error.size else 0.0
                max_abs_xy_error = max(max_abs_xy_error, frame_max_error)
                frames_compared_to_source += 1
                frame_report["max_source_xy_error_m"] = frame_max_error

                if frame_max_error > args.atol:
                    raise AssertionError(
                        f"Frame {idx}: source/stream xy mismatch {frame_max_error:.6f} > {args.atol}"
                    )
            elif planning is not None and args.strict_missing:
                raise AssertionError(f"Frame {idx}: stream exists but source prediction is missing")

        per_frame.append(frame_report)

    report = {
        "scene_dir": str(scene_dir),
        "frame_count": len(messages),
        "frames_with_planning_stream": frames_with_stream,
        "frames_missing_planning_stream": frames_missing_stream,
        "frames_compared_to_source": frames_compared_to_source,
        "max_abs_source_xy_error_m": max_abs_xy_error,
        "max_first_point_ego_error_m": max(first_point_errors) if first_point_errors else None,
        "per_frame": per_frame,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SparseDrive final_planning inside NUSVIZ /ego/planning_trajectory."
    )
    parser.add_argument("--scene-dir", required=True, help="NUSVIZ scene directory")
    parser.add_argument("--sparsedrive-pkl", default=None, help="SparseDrive prediction pkl for source comparison")
    parser.add_argument("--nuscenes-dataroot", default=None, help="nuScenes dataroot, required with --sparsedrive-pkl")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance in meters")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if a planning stream exists for a frame missing from the SparseDrive pkl",
    )
    parser.add_argument("--report-json", default=None, help="Optional path to write a JSON report")

    args = parser.parse_args()
    report = validate_scene(args)

    print(json.dumps({k: v for k, v in report.items() if k != "per_frame"}, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
