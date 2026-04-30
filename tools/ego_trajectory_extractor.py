"""
nuScenes 自车轨迹提取工具

功能：
- 基于 nuscenes-devkit 提取训练集(train)与验证集(val)的自车(ego)轨迹
- 每个场景输出：场景名称(scene_name)、时间戳(timestamp, 微秒)、轨迹路径(ego_pose.translation, 全局坐标系)

输出：
- tools/ego_trajectories/ego_trajectories_train.json
- tools/ego_trajectories/ego_trajectories_val.json

硬编码配置：
- 数据根目录：/home/public/nuscenes_datasets/nuscenes-trainval
- 版本：v1.0-trainval
- 输出目录：脚本所在目录下的 ego_trajectories 子目录

使用方法：
  python tools/ego_trajectory_extractor.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


from tqdm import tqdm



# ===== 硬编码配置 =====
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_DATAROOT = "/home/public/nuscenes_datasets/nuscenes-trainval"
OUTPUT_SUBDIR = "ego_trajectories"  # 相对于脚本所在目录的子目录
JSON_INDENT = 2
SHOW_PROGRESS = True


@dataclass(frozen=True)
class SceneEgoTrajectory:
    """单个场景的自车轨迹数据（可直接序列化为JSON）。"""

    scene_token: str
    scene_name: str
    location: str  # 城市名称，如 singapore-onenorth, boston-seaport
    timestamps: List[int]  # microseconds
    trajectory: List[List[float]]  # [x, y, z] in global frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_token": self.scene_token,
            "scene_name": self.scene_name,
            "timestamps": self.timestamps,
            "trajectory": self.trajectory,
        }


def get_train_val_scene_names() -> Tuple[List[str], List[str]]:
    """获取 v1.0-trainval 的 train/val scene name 列表。"""
    return list(splits.train), list(splits.val)


def build_scene_name_to_token(nusc: NuScenes) -> Dict[str, str]:
    return {scene["name"]: scene["token"] for scene in nusc.scene}


def extract_scene_ego_trajectory(
    nusc: NuScenes, scene_token: str
) -> SceneEgoTrajectory:
    """
    提取单个场景的 ego 轨迹：
    - 时间戳：sample['timestamp']（keyframe 时间，微秒）
    - 轨迹点：对应 keyframe 的 LIDAR_TOP sample_data -> ego_pose.translation（全局坐标系）
    - 城市名称：scene -> log -> location
    """
    scene = nusc.get("scene", scene_token)
    scene_name = scene["name"]
    
    # 提取城市名称
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    timestamps: List[int] = []
    trajectory: List[List[float]] = []

    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        timestamps.append(int(sample["timestamp"]))

        lidar_sd_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_sd_token)
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        translation = [float(x) for x in ego_pose["translation"]]
        trajectory.append(translation)

        sample_token = sample["next"]

    return SceneEgoTrajectory(
        scene_token=scene_token,
        scene_name=scene_name,
        location=location,
        timestamps=timestamps,
        trajectory=trajectory,
    )


def extract_split_ego_trajectories(
    nusc: NuScenes,
    split_scene_names: List[str],
    split_name: str,
    *,
    show_progress: bool = True,
) -> List[SceneEgoTrajectory]:
    name_to_token = build_scene_name_to_token(nusc)

    trajectories: List[SceneEgoTrajectory] = []

    iterator = split_scene_names
    if show_progress:
        iterator = tqdm(
            split_scene_names, desc=f"Extracting {split_name}", unit="scene"
        )

    missing: List[str] = []
    for scene_name in iterator:
        scene_token = name_to_token.get(scene_name)
        if not scene_token:
            missing.append(scene_name)
            continue
        trajectories.append(extract_scene_ego_trajectory(nusc, scene_token))

    if missing:
        print(
            f"[WARN] {split_name}: {len(missing)} scene names not found in current version. "
            f"Example: {missing[0]}"
        )

    return trajectories


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """写入 JSON 文件（自动覆盖已有文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def build_output_payload(
    *,
    split_name: str,
    trajectories: List[SceneEgoTrajectory],
) -> Dict[str, Any]:
    """构建输出数据，将场景按城市分组。"""
    # 按城市分组
    scenes_by_location: Dict[str, List[Dict[str, Any]]] = {}
    for traj in trajectories:
        location = traj.location
        if location not in scenes_by_location:
            scenes_by_location[location] = []
        scenes_by_location[location].append(traj.to_dict())
    
    return {
        "split": split_name,
        "num_scenes": len(trajectories),
        "scenes": scenes_by_location,
    }


def main() -> None:
    """主函数：使用硬编码配置提取自车轨迹。"""
    # 硬编码配置
    dataroot = NUSCENES_DATAROOT
    version = NUSCENES_VERSION
    output_dir = Path(__file__).parent / OUTPUT_SUBDIR

    # 验证数据根目录
    dataroot_path = Path(dataroot)
    if not dataroot_path.exists():
        raise FileNotFoundError(f"数据根目录不存在: {dataroot_path}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 train/val 场景列表
    train_scene_names, val_scene_names = get_train_val_scene_names()

    print("=" * 80)
    print("nuScenes 自车轨迹提取工具")
    print(f"版本          : {version}")
    print(f"数据根目录    : {dataroot}")
    print(f"输出目录      : {output_dir}")
    print(f"train 场景数  : {len(train_scene_names)}")
    print(f"val 场景数    : {len(val_scene_names)}")
    print("=" * 80)

    # 初始化 NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # 提取 train 和 val 轨迹
    train_trajs = extract_split_ego_trajectories(
        nusc, train_scene_names, "train", show_progress=SHOW_PROGRESS
    )
    val_trajs = extract_split_ego_trajectories(
        nusc, val_scene_names, "val", show_progress=SHOW_PROGRESS
    )

    # 构建输出数据
    train_payload = build_output_payload(split_name="train", trajectories=train_trajs)
    val_payload = build_output_payload(split_name="val", trajectories=val_trajs)

    # 保存 JSON 文件
    train_out = output_dir / "ego_trajectories_train.json"
    val_out = output_dir / "ego_trajectories_val.json"

    write_json(train_out, train_payload, indent=JSON_INDENT)
    write_json(val_out, val_payload, indent=JSON_INDENT)

    print()
    print(f"✓ 已保存 train 轨迹: {train_out} ({len(train_trajs)} 个场景)")
    print(f"✓ 已保存 val   轨迹: {val_out} ({len(val_trajs)} 个场景)")
    print("=" * 80)


if __name__ == "__main__":
    main()
