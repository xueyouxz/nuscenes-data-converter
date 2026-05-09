"""
Export nuScenes train+val scene inventory and ego trajectories.

Outputs:
- Scene inventory JSON: scene description, scene name, map name, and unique object
  counts by raw nuScenes category for every train/val scene.
- Ego trajectory JSON: global ego trajectory for every train/val scene.

Usage:
  python tools/export_nuscenes_scene_inventory.py

Optional:
  python tools/export_nuscenes_scene_inventory.py \
    --dataroot /home/public/nuscenes_datasets/nuscenes-trainval \
    --version v1.0-trainval \
    --scene-output tools/nuscenes_scene_object_summary.json \
    --trajectory-output tools/ego_trajectories/ego_trajectories_trainval.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from tqdm import tqdm


DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_DATAROOT = "/home/public/nuscenes_datasets/nuscenes-trainval"
DEFAULT_SCENE_OUTPUT = "tools/nuscenes_scene_object_summary.json"
DEFAULT_TRAJECTORY_OUTPUT = "tools/ego_trajectories/ego_trajectories_trainval.json"
JSON_INDENT = 2


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=JSON_INDENT)


def scene_name_to_record(nusc: NuScenes) -> Dict[str, Dict[str, Any]]:
    return {scene["name"]: scene for scene in nusc.scene}


def split_scene_names() -> Dict[str, List[str]]:
    return {
        "train": list(splits.train),
        "val": list(splits.val),
    }


def iter_scene_samples(nusc: NuScenes, scene: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        yield sample
        sample_token = sample["next"]


def build_log_token_to_map(nusc: NuScenes) -> Dict[str, Dict[str, Any]]:
    """Map every nuScenes log token to its map record."""
    mapping: Dict[str, Dict[str, Any]] = {}
    for map_record in nusc.map:
        for log_token in map_record.get("log_tokens", []):
            mapping[log_token] = map_record
    return mapping


def map_name_from_record(map_record: Optional[Dict[str, Any]], fallback_location: str) -> str:
    if not map_record:
        return fallback_location

    filename = map_record.get("filename")
    if filename:
        return Path(filename).stem

    return map_record.get("category") or fallback_location


def count_unique_objects_by_category(
    nusc: NuScenes, scene: Dict[str, Any]
) -> Tuple[Dict[str, int], int]:
    """Count unique instance tokens per raw nuScenes annotation category."""
    category_to_instances: Dict[str, Set[str]] = defaultdict(set)

    for sample in iter_scene_samples(nusc, scene):
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category_to_instances[ann["category_name"]].add(ann["instance_token"])

    counts = {
        category: len(instances)
        for category, instances in sorted(category_to_instances.items())
    }
    unique_total = len(set().union(*category_to_instances.values())) if counts else 0
    return counts, unique_total


def extract_ego_trajectory(nusc: NuScenes, scene: Dict[str, Any]) -> Dict[str, Any]:
    timestamps: List[int] = []
    sample_tokens: List[str] = []
    trajectory: List[List[float]] = []
    poses: List[Dict[str, Any]] = []

    for sample in iter_scene_samples(nusc, scene):
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        rotation = Quaternion(ego_pose["rotation"])

        timestamp = int(sample["timestamp"])
        translation = [float(value) for value in ego_pose["translation"]]
        rotation_values = [float(value) for value in ego_pose["rotation"]]

        timestamps.append(timestamp)
        sample_tokens.append(sample["token"])
        trajectory.append(translation)
        poses.append(
            {
                "sample_token": sample["token"],
                "timestamp": timestamp,
                "translation": translation,
                "rotation": rotation_values,
                "yaw": float(quaternion_yaw(rotation)),
            }
        )

    return {
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "timestamps": timestamps,
        "sample_tokens": sample_tokens,
        "trajectory": trajectory,
        "poses": poses,
    }


def build_scene_inventory_entry(
    nusc: NuScenes,
    scene: Dict[str, Any],
    split_name: str,
    log_token_to_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    log = nusc.get("log", scene["log_token"])
    map_record = log_token_to_map.get(scene["log_token"])
    object_counts, object_total = count_unique_objects_by_category(nusc, scene)

    return {
        "split": split_name,
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "map_name": map_name_from_record(map_record, log["location"]),
        "map_filename": map_record.get("filename") if map_record else None,
        "location": log["location"],
        "nbr_samples": int(scene["nbr_samples"]),
        "object_total_unique": object_total,
        "object_counts_by_category": object_counts,
    }


def export_scene_inventory_and_trajectories(
    nusc: NuScenes,
    *,
    show_progress: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scene_by_name = scene_name_to_record(nusc)
    log_token_to_map = build_log_token_to_map(nusc)
    requested_splits = split_scene_names()

    inventory_scenes: List[Dict[str, Any]] = []
    trajectory_scenes: List[Dict[str, Any]] = []
    missing: Dict[str, List[str]] = {"train": [], "val": []}

    for split_name, names in requested_splits.items():
        iterator: Iterable[str] = names
        if show_progress:
            iterator = tqdm(names, desc=f"Exporting {split_name}", unit="scene")

        for scene_name in iterator:
            scene = scene_by_name.get(scene_name)
            if scene is None:
                missing[split_name].append(scene_name)
                continue

            inventory_scenes.append(
                build_scene_inventory_entry(nusc, scene, split_name, log_token_to_map)
            )
            trajectory_scenes.append(extract_ego_trajectory(nusc, scene))

    split_counts = {
        split_name: sum(1 for scene in inventory_scenes if scene["split"] == split_name)
        for split_name in requested_splits
    }

    scene_inventory = {
        "version": nusc.version,
        "dataroot": str(nusc.dataroot),
        "summary": {
            "total_scenes": len(inventory_scenes),
            "splits": split_counts,
            "missing_scenes": {key: value for key, value in missing.items() if value},
        },
        "scenes": inventory_scenes,
    }

    ego_trajectories = {
        "version": nusc.version,
        "dataroot": str(nusc.dataroot),
        "summary": {
            "total_scenes": len(trajectory_scenes),
            "splits": split_counts,
        },
        "scenes": trajectory_scenes,
    }

    return scene_inventory, ego_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export train+val nuScenes scene inventory and global ego trajectories."
    )
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT, help="nuScenes dataroot")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="nuScenes version")
    parser.add_argument(
        "--scene-output",
        default=DEFAULT_SCENE_OUTPUT,
        help="Output JSON path for scene inventory",
    )
    parser.add_argument(
        "--trajectory-output",
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Output JSON path for global ego trajectories",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        raise FileNotFoundError(f"nuScenes dataroot does not exist: {dataroot}")

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    scene_inventory, ego_trajectories = export_scene_inventory_and_trajectories(
        nusc, show_progress=not args.no_progress
    )

    scene_output = Path(args.scene_output)
    trajectory_output = Path(args.trajectory_output)
    write_json(scene_output, scene_inventory)
    write_json(trajectory_output, ego_trajectories)

    print(f"Saved scene inventory: {scene_output}")
    print(f"Saved ego trajectories: {trajectory_output}")
    print(
        "Exported "
        f"{scene_inventory['summary']['splits'].get('train', 0)} train scenes and "
        f"{scene_inventory['summary']['splits'].get('val', 0)} val scenes."
    )


if __name__ == "__main__":
    main()
