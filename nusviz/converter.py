import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .metadata_builder import MetadataBuilder
    from .message_builder import MessageBuilder
except ImportError:
    from metadata_builder import MetadataBuilder
    from message_builder import MessageBuilder


# ===== 硬编码配置 =====
NUSCENES_VERSION  = "v1.0-trainval"
NUSCENES_DATAROOT = "/home/public/nuscenes_datasets/nuscenes-trainval"
OUTPUT_ROOT       = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/nusviz/output"
SPARSEDRIVE_PREDICTION = str(
    _REPO_ROOT / "data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl"
)
DEFAULT_SPLIT     = "val"

_SPLIT_SCENE_NAMES = {
    'train':      splits.train,
    'val':        splits.val,
    'test':       splits.test,
    'mini_train': splits.mini_train,
    'mini_val':   splits.mini_val,
}


class _NuVizEgoPoseAdapter:
    """为 SparseDriveExtractor 提供所需的 extract_ego_pose 接口。"""

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def extract_ego_pose(self, sample_token: str) -> Dict[str, object]:
        from nuscenes.eval.common.utils import quaternion_yaw

        sample = self.nusc.get('sample', sample_token)
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        rotation = Quaternion(ego_pose['rotation'])

        return {
            'sample_token': sample_token,
            'translation': ego_pose['translation'],
            'rotation': rotation.q.tolist(),
            'yaw': float(quaternion_yaw(rotation)),
            'timestamp': sample['timestamp'],
        }


class NuScenesConverter:
    """将 NuScenes 数据集转换为 nuviz 格式。"""

    def __init__(
        self,
        dataroot: str = NUSCENES_DATAROOT,
        version: str = NUSCENES_VERSION,
        output_root: str = OUTPUT_ROOT,
        sparsedrive_prediction: Optional[str] = SPARSEDRIVE_PREDICTION,
    ):
        self.dataroot    = dataroot
        self.version     = version
        self.output_root = Path(output_root)
        self.sparsedrive_prediction = sparsedrive_prediction

        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        print(f"Loaded {len(self.nusc.scene)} scenes")

        self.sd_extractor = None
        if sparsedrive_prediction:
            prediction_path = Path(sparsedrive_prediction)
            if not prediction_path.exists():
                raise FileNotFoundError(f"SparseDrive prediction file not found: {sparsedrive_prediction}")

            from data_converter.core.sparsedrive_extractor import SparseDriveExtractor

            print(f"Loading SparseDrive predictions from {sparsedrive_prediction}...")
            self.sd_extractor = SparseDriveExtractor(
                sparsedrive_prediction,
                _NuVizEgoPoseAdapter(self.nusc),
            )

    def convert_split(self, split: str = DEFAULT_SPLIT):
        """
        转换指定 split 的所有场景。

        split 可选值：'train', 'val', 'test', 'mini_train', 'mini_val'
        """
        scene_names = _SPLIT_SCENE_NAMES.get(split)
        if scene_names is None:
            raise ValueError(f"Unknown split: {split}")

        scenes = [s for s in self.nusc.scene if s['name'] in scene_names]

        print(f"\n{'='*80}")
        print(f"Converting {len(scenes)} scenes from split '{split}'")
        print(f"{'='*80}\n")

        for scene in tqdm(scenes, desc=f"Converting {split}", unit="scene"):
            self.convert_scene(scene['token'])

    def convert_scene_name(self, scene_name: str):
        """按 scene name 转换单个场景。"""
        for scene in self.nusc.scene:
            if scene['name'] == scene_name:
                return self.convert_scene(scene['token'])
        raise ValueError(f"Unknown scene name: {scene_name}")

    def convert_scene(self, scene_token: str):
        """转换单个场景，输出 metadata.glb、messages/*.glb 和 message_index.json。"""
        scene      = self.nusc.get('scene', scene_token)
        scene_name = scene['name']
        log_rec    = self.nusc.get('log', scene['log_token'])

        scene_dir    = self.output_root / scene_name
        messages_dir = scene_dir / "messages"

        all_samples = self._collect_samples(scene_token)
        sample_to_original_idx = {token: idx for idx, token in enumerate(all_samples)}
        samples = self._select_convertible_samples(all_samples)
        if not samples:
            print(f"Skipping {scene_name}: no keyframes with SparseDrive predictions")
            return None

        messages_dir.mkdir(parents=True, exist_ok=True)
        for stale_message in messages_dir.glob("*.glb"):
            stale_message.unlink()

        # 1. metadata.glb
        (scene_dir / "metadata.glb").write_bytes(
            MetadataBuilder(
                self.nusc,
                scene_token,
                dataroot=self.dataroot,
                sample_tokens=samples,
            ).build()
        )

        # 2. 预计算全场景轨迹数据。消息只写有预测的关键帧，但 GT future
        #    仍使用原始 scene 时间线，保证末尾预测帧能看到后续真实轨迹。
        ego_all  = self._collect_ego_poses(all_samples)         # list of [x,y,z], len=N
        obj_all  = self._collect_obj_trajectories(all_samples)  # track_id -> [(frame_idx,[x,y,z]),...]

        # 3. messages/*.glb
        message_builder = MessageBuilder(self.nusc, self.dataroot)
        message_entries = []
        prediction_frame_count = 0

        for idx, sample_token in enumerate(samples):
            sample    = self.nusc.get('sample', sample_token)
            timestamp = sample['timestamp'] / 1e6
            original_idx = sample_to_original_idx[sample_token]

            # 自车未来轨迹：从当前帧（含）到末帧
            ego_future = ego_all[original_idx:]

            # 对象未来轨迹：与当前帧 anns 顺序严格对应
            obj_future = self._get_obj_future_for_frame(sample, original_idx, obj_all)

            planning_trajectory = self._get_planning_trajectory(sample_token)
            sparsedrive_detections = self._get_sparsedrive_detections(sample_token)
            sparsedrive_map_predictions = self._get_sparsedrive_map_predictions(sample_token)
            if self.sd_extractor is not None:
                prediction_frame_count += 1

            update_type = "COMPLETE_STATE" if idx == 0 else "INCREMENTAL"
            msg_bytes   = message_builder.build_message(
                sample_token,
                update_type=update_type,
                include_lidar=True,
                include_cameras=True,
                include_objects=True,
                ego_future_poses=ego_future,
                planning_trajectory=planning_trajectory,
                obj_future_trajectories=obj_future,
                sparsedrive_detections=sparsedrive_detections,
                sparsedrive_map_predictions=sparsedrive_map_predictions,
            )

            filename = f"{idx:06d}.glb"
            (messages_dir / filename).write_bytes(msg_bytes)
            message_entries.append({
                "index":     idx,
                "timestamp": timestamp,
                "file":      f"messages/{filename}",
                "extensions": {
                    "nuscenes": {
                        "sample_token": sample_token,
                        "original_sample_index": original_idx,
                    }
                },
            })

        # 4. message_index.json
        first_sample = self.nusc.get('sample', samples[0])
        last_sample  = self.nusc.get('sample', samples[-1])

        message_index = {
            "message_format": "BINARY",
            "metadata": "metadata.glb",
            "log_info": {
                "start_time": first_sample['timestamp'] / 1e6,
                "end_time":   last_sample['timestamp']  / 1e6,
            },
            "messages": message_entries,
            "extensions": {
                "nuscenes": {
                    "scene_token": scene_token,
                    "scene_name":  scene_name,
                    "mapId":       log_rec['location'],
                }
            },
        }

        (scene_dir / "message_index.json").write_text(
            json.dumps(message_index, indent=2)
        )

        if self.sd_extractor is not None:
            print(
                f"Converted {prediction_frame_count}/{len(all_samples)} keyframes "
                f"with SparseDrive predictions in {scene_name}"
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _collect_samples(self, scene_token: str) -> List[str]:
        """按时间顺序收集场景内所有 sample token。"""
        scene  = self.nusc.get('scene', scene_token)
        tokens = []
        token  = scene['first_sample_token']
        while token:
            tokens.append(token)
            token = self.nusc.get('sample', token)['next']
        return tokens

    def _select_convertible_samples(self, sample_tokens: List[str]) -> List[str]:
        """
        V2 预测协议只写包含模型预测结果的关键帧。

        没有配置 SparseDrive 预测文件时保留所有 sample，便于调试纯 GT 输出；
        配置了预测文件时严格按 sample_token 过滤，场景尾部缺预测的帧不会生成
        message 文件。
        """
        if self.sd_extractor is None:
            return sample_tokens
        return [
            sample_token
            for sample_token in sample_tokens
            if self.sd_extractor.has_prediction(sample_token)
        ]

    def _get_planning_trajectory(self, sample_token: str) -> Optional[List[List[float]]]:
        """
        提取 SparseDrive final_planning 并补齐为 NUSVIZ VEC3 轨迹。

        Returns:
            list of [x, y, z]，世界坐标系；缺预测时返回 None。
        """
        if self.sd_extractor is None:
            return None
        if not self.sd_extractor.has_prediction(sample_token):
            return None

        planning_xy = self.sd_extractor.extract_planning(sample_token)
        if not planning_xy:
            return None

        return [[point[0], point[1], 0.0] for point in planning_xy]

    def _get_sparsedrive_detections(self, sample_token: str) -> Optional[List[Dict[str, object]]]:
        if self.sd_extractor is None:
            return None
        if not self.sd_extractor.has_prediction(sample_token):
            return None
        return self.sd_extractor.extract_detections(sample_token)

    def _get_sparsedrive_map_predictions(self, sample_token: str) -> Optional[List[Dict[str, object]]]:
        if self.sd_extractor is None:
            return None
        if not self.sd_extractor.has_prediction(sample_token):
            return None
        return self.sd_extractor.extract_map_predictions(sample_token)

    def _collect_ego_poses(self, sample_tokens: List[str]) -> List[List[float]]:
        """
        按帧顺序收集自车在世界坐标系中的位置（translation [x,y,z]）。

        Returns:
            list of [x, y, z]，长度 = 总帧数 N
        """
        poses = []
        for token in sample_tokens:
            sample   = self.nusc.get('sample', token)
            lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ep       = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            poses.append(list(ep['translation']))
        return poses

    def _collect_obj_trajectories(
        self, sample_tokens: List[str]
    ) -> Dict[int, List[Tuple[int, List[float]]]]:
        """
        扫描场景所有帧，构建 track_id -> [(frame_idx, [x,y,z]), ...] 映射。

        用于后续按帧、按 track_id 快速查询对象的未来轨迹点。
        每条记录按 frame_idx 升序排列（遍历顺序保证有序）。
        """
        traj: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)

        for frame_idx, token in enumerate(sample_tokens):
            sample = self.nusc.get('sample', token)
            for ann_token in sample['anns']:
                ann      = self.nusc.get('sample_annotation', ann_token)
                track_id = self.nusc.getind('instance', ann['instance_token'])
                traj[track_id].append((frame_idx, list(ann['translation'])))

        return traj

    def _get_obj_future_for_frame(
        self,
        sample: dict,
        current_frame_idx: int,
        obj_all: Dict[int, List[Tuple[int, List[float]]]],
    ) -> List[List[List[float]]]:
        """
        为当前帧的每个标注对象构建其未来轨迹点列表，顺序与 sample['anns'] 严格对应。

        Args:
            sample:            当前帧的 nuScenes sample 记录
            current_frame_idx: 当前帧在场景中的帧序号（0-based）
            obj_all:           _collect_obj_trajectories 返回的全场景轨迹字典

        Returns:
            list of list of [x,y,z]，长度 = len(sample['anns'])；
            第 i 项为第 i 个标注对象从当前帧（含）往后所有出现帧的中心点序列。
            若某对象在当前帧之后不再出现，则对应项为单元素列表（仅含当前帧位置）。
        """
        result: List[List[List[float]]] = []

        for ann_token in sample['anns']:
            ann      = self.nusc.get('sample_annotation', ann_token)
            track_id = self.nusc.getind('instance', ann['instance_token'])

            # 取该 track 中 frame_idx >= current_frame_idx 的所有点
            future_points = [
                xyz
                for (fidx, xyz) in obj_all.get(track_id, [])
                if fidx >= current_frame_idx
            ]

            result.append(future_points)

        return result


def _default_prediction_path() -> Optional[str]:
    """Return the repo-local SparseDrive pkl if it exists; otherwise keep planning disabled."""
    path = Path(SPARSEDRIVE_PREDICTION)
    return str(path) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Convert nuScenes scenes to NUSVIZ GLB format.")
    parser.add_argument("--dataroot", default=NUSCENES_DATAROOT, help="nuScenes dataset root")
    parser.add_argument("--version", default=NUSCENES_VERSION, help="nuScenes version")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="NUSVIZ output root")
    parser.add_argument("--split", default=None, help="Split to convert, e.g. val or mini_val")
    parser.add_argument("--scene-name", default=None, help="Convert a single scene by name, e.g. scene-0916")
    parser.add_argument(
        "--sparsedrive-prediction",
        default=_default_prediction_path(),
        help=(
            "SparseDrive pkl path. Defaults to the repo-local data/sparsedrive pkl "
            "when it exists; pass an empty string to disable planning."
        ),
    )
    args = parser.parse_args()

    sparsedrive_prediction = args.sparsedrive_prediction or None

    converter = NuScenesConverter(
        dataroot=args.dataroot,
        version=args.version,
        output_root=args.output_root,
        sparsedrive_prediction=sparsedrive_prediction,
    )

    if args.scene_name:
        converter.convert_scene_name(args.scene_name)
    else:
        converter.convert_split(args.split or DEFAULT_SPLIT)

    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"Output directory: {args.output_root}")
    print(f"SparseDrive planning: {'enabled' if sparsedrive_prediction else 'disabled'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
