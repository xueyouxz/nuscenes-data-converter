"""
数据转换器使用示例
"""

from data_converter import DataConverter
from typing import List


def get_dataroot_by_version(version: str) -> str:
    """根据版本获取数据集根目录"""
    version_to_dataroot = {
        'v1.0-mini': '/home/public/nuscenes_datasets/nuscenes-mini',
        'v1.0-trainval': '/home/public/nuscenes_datasets/nuscenes-trainval',
    }
    return version_to_dataroot[version]


def convert_single_scene(
        scene_name: str,
        version: str,
        sparsedrive_prediction: str,
        output_dir: str
):
    """转换单个场景"""
    dataroot = get_dataroot_by_version(version)
    
    converter = DataConverter(
        nuscenes_dataroot=dataroot,
        sparsedrive_prediction=sparsedrive_prediction,
        nuscenes_version=version
    )
    
    result = converter.convert_scene(scene_name, output_dir)
    
    print(f"场景名称: {result['scene_name']}")
    print("生成的文件:")
    for file_type, file_path in result['files'].items():
        print(f"  - {file_type}: {file_path}")


def convert_multiple_scenes(
        scene_names: List[str],
        version: str,
        sparsedrive_prediction: str,
        output_dir: str,
        max_workers: int = 4
):
    """批量转换多个场景（多线程）"""
    dataroot = get_dataroot_by_version(version)
    
    converter = DataConverter(
        nuscenes_dataroot=dataroot,
        sparsedrive_prediction=sparsedrive_prediction,
        nuscenes_version=version
    )
    
    result = converter.convert_scenes(
        scene_names=scene_names,
        output_dir=output_dir,
        max_workers=max_workers
    )
    
    print(f"批量转换完成！共转换 {len(result['scenes'])} 个场景")
    print(f"索引文件: {result['index_file']}")


if __name__ == '__main__':
    sparsedrive_prediction = 'data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl'
    output_dir = './output/scenes'
    
    from nuscenes.utils import splits
    scene_name_list = splits.mini_val
    
    # 示例1: 转换单个场景
    # convert_single_scene(
    #     scene_name=scene_name_list[0],
    #     version='v1.0-mini',
    #     sparsedrive_prediction=sparsedrive_prediction,
    #     output_dir=output_dir
    # )
    
    # 示例2: 批量转换多个场景（多线程）
    convert_multiple_scenes(
        scene_names=splits.val,
        version='v1.0-trainval',
        sparsedrive_prediction=sparsedrive_prediction,
        output_dir=output_dir,
        max_workers=8
    )

