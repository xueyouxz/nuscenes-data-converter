import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from metadata_builder import MetadataBuilder
from message_builder import MessageBuilder


# ===== 硬编码配置 =====
NUSCENES_VERSION  = "v1.0-trainval"
NUSCENES_DATAROOT = "/home/public/nuscenes_datasets/nuscenes-trainval"
OUTPUT_ROOT       = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/nusviz/output"
DEFAULT_SPLIT     = "val"

_SPLIT_SCENE_NAMES = {
    'train':      splits.train,
    'val':        splits.val,
    'test':       splits.test,
    'mini_train': splits.mini_train,
    'mini_val':   splits.mini_val,
}


class NuScenesConverter:
    """将 NuScenes 数据集转换为 nuviz 格式。"""

    def __init__(
        self,
        dataroot: str = NUSCENES_DATAROOT,
        version: str = NUSCENES_VERSION,
        output_root: str = OUTPUT_ROOT,
    ):
        self.dataroot    = dataroot
        self.version     = version
        self.output_root = Path(output_root)

        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        print(f"Loaded {len(self.nusc.scene)} scenes")

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

    def convert_scene(self, scene_token: str):
        """转换单个场景，输出 metadata.glb、messages/*.glb 和 message_index.json。"""
        scene      = self.nusc.get('scene', scene_token)
        scene_name = scene['name']
        log_rec    = self.nusc.get('log', scene['log_token'])

        scene_dir    = self.output_root / scene_name
        messages_dir = scene_dir / "messages"
        messages_dir.mkdir(parents=True, exist_ok=True)

        # 1. metadata.glb
        (scene_dir / "metadata.glb").write_bytes(
            MetadataBuilder(self.nusc, scene_token).build()
        )

        # 2. 预计算全场景轨迹数据
        samples  = self._collect_samples(scene_token)
        ego_all  = self._collect_ego_poses(samples)         # list of [x,y,z], len=N
        obj_all  = self._collect_obj_trajectories(samples)  # track_id -> [(frame_idx,[x,y,z]),...]

        # 3. messages/*.glb
        message_builder = MessageBuilder(self.nusc, self.dataroot)
        message_entries = []

        for idx, sample_token in enumerate(samples):
            sample    = self.nusc.get('sample', sample_token)
            timestamp = sample['timestamp'] / 1e6

            # 自车未来轨迹：从当前帧（含）到末帧
            ego_future = ego_all[idx:]

            # 对象未来轨迹：与当前帧 anns 顺序严格对应
            obj_future = self._get_obj_future_for_frame(sample, idx, obj_all)

            update_type = "COMPLETE_STATE" if idx == 0 else "INCREMENTAL"
            msg_bytes   = message_builder.build_message(
                sample_token,
                update_type=update_type,
                include_lidar=True,
                include_cameras=True,
                include_objects=True,
                ego_future_poses=ego_future,
                obj_future_trajectories=obj_future,
            )

            filename = f"{idx:06d}.glb"
            (messages_dir / filename).write_bytes(msg_bytes)
            message_entries.append({
                "index":     idx,
                "timestamp": timestamp,
                "file":      f"messages/{filename}",
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


if __name__ == '__main__':
    converter = NuScenesConverter(
        dataroot=NUSCENES_DATAROOT,
        version=NUSCENES_VERSION,
        output_root=OUTPUT_ROOT,
    )
    SPLIT = "mini_val"
    converter.convert_split(SPLIT)

    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"{'='*80}\n")
