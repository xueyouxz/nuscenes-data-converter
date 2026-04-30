"""
nuScenes 场景元数据生成工具

功能：
- 基于 nuscenes-devkit 生成场景元数据文件 nuscenes.json
- 按城市名称分组，区分训练集和验证集
- 提取每个场景的基本信息：scene_name, scene_token, scene_description, frame_count

输出：
- tools/nuscenes.json

硬编码配置：
- 数据根目录：/home/public/nuscenes_datasets/nuscenes-trainval
- 版本：v1.0-trainval

使用方法：
  python tools/generate_nuscenes_metadata.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


# ===== 硬编码配置 =====
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_DATAROOT = "/home/public/nuscenes_datasets/nuscenes-trainval"
OUTPUT_FILE = "nuscenes.json"  # 输出文件名
JSON_INDENT = 2


def count_frames_in_scene(nusc: NuScenes, scene_token: str) -> int:
    """统计场景中的帧数（sample数量）"""
    scene = nusc.get('scene', scene_token)
    return scene['nbr_samples']


def extract_scene_metadata(nusc: NuScenes, scene_token: str) -> dict:
    """提取单个场景的元数据"""
    scene = nusc.get('scene', scene_token)
    
    return {
        'scene_name': scene['name'],
        'scene_token': scene_token,
        'scene_description': scene['description'],
        'frame_count': scene['nbr_samples']
    }


def generate_metadata(nusc: NuScenes) -> dict:
    """
    生成完整的元数据字典，按城市分组
    
    返回结构：
    {
        "singapore-queenstown": {
            "val": [...],
            "train": [...]
        },
        ...
    }
    """
    # 获取训练集和验证集的场景名称列表
    train_scenes = splits.train
    val_scenes = splits.val
    
    # 按城市分组
    metadata_by_location = defaultdict(lambda: {"train": [], "val": []})
    
    print("处理验证集场景...")
    for scene_name in tqdm(val_scenes, desc="验证集"):
        # 通过 scene_name 查找 scene
        scene_record = None
        for scene in nusc.scene:
            if scene['name'] == scene_name:
                scene_record = scene
                break
        
        if scene_record is None:
            print(f"警告: 未找到场景 {scene_name}")
            continue
        
        # 获取场景的位置信息
        first_sample_token = scene_record['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        log = nusc.get('log', nusc.get('scene', scene_record['token'])['log_token'])
        location = log['location']
        
        # 提取场景元数据
        scene_metadata = extract_scene_metadata(nusc, scene_record['token'])
        metadata_by_location[location]['val'].append(scene_metadata)
    
    print("\n处理训练集场景...")
    for scene_name in tqdm(train_scenes, desc="训练集"):
        # 通过 scene_name 查找 scene
        scene_record = None
        for scene in nusc.scene:
            if scene['name'] == scene_name:
                scene_record = scene
                break
        
        if scene_record is None:
            print(f"警告: 未找到场景 {scene_name}")
            continue
        
        # 获取场景的位置信息
        first_sample_token = scene_record['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        log = nusc.get('log', nusc.get('scene', scene_record['token'])['log_token'])
        location = log['location']
        
        # 提取场景元数据
        scene_metadata = extract_scene_metadata(nusc, scene_record['token'])
        metadata_by_location[location]['train'].append(scene_metadata)
    
    # 转换 defaultdict 为普通 dict
    return dict(metadata_by_location)


def generate_summary(metadata: dict) -> dict:
    """生成统计摘要"""
    summary = {
        "total_train": 0,
        "total_val": 0,
        "locations": {}
    }
    
    for location, splits_data in metadata.items():
        train_count = len(splits_data['train'])
        val_count = len(splits_data['val'])
        
        summary['total_train'] += train_count
        summary['total_val'] += val_count
        summary['locations'][location] = {
            'train': train_count,
            'val': val_count
        }
    
    return summary


def main():
    """主函数"""
    # 确定输出路径
    script_dir = Path(__file__).parent
    output_path = script_dir / OUTPUT_FILE
    
    print(f"nuScenes 场景元数据生成工具")
    print(f"数据根目录: {NUSCENES_DATAROOT}")
    print(f"版本: {NUSCENES_VERSION}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 加载 nuScenes 数据集
    print("\n加载 nuScenes 数据集...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    
    # 生成元数据
    print("\n生成场景元数据...")
    metadata = generate_metadata(nusc)
    
    # 生成统计摘要
    summary = generate_summary(metadata)
    
    # 构建最终输出结构
    output_data = {
        "summary": summary,
        "scenes": metadata
    }
    
    # 保存到文件
    print(f"\n保存元数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=JSON_INDENT, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  训练集场景总数: {summary['total_train']}")
    print(f"  验证集场景总数: {summary['total_val']}")
    print(f"  场景总数: {summary['total_train'] + summary['total_val']}")
    print("\n按城市分布:")
    for location, counts in summary['locations'].items():
        print(f"  {location}:")
        print(f"    训练集: {counts['train']}")
        print(f"    验证集: {counts['val']}")
    print("=" * 60)
    print(f"\n✓ 元数据文件已生成: {output_path}")


if __name__ == "__main__":
    main()

