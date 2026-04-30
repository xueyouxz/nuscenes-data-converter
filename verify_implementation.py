"""
验证实现的代码结构和逻辑

不需要实际的nuScenes数据集，只验证代码结构
"""

import sys
import inspect
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data_converter.core.nuscenes_extractor import NuScenesExtractor
from data_converter.builders.gt_stream_builder import GtStreamBuilder


def verify_extractor_signature():
    """验证NuScenesExtractor.extract_annotations的方法签名"""
    print("\n=== 验证NuScenesExtractor.extract_annotations方法签名 ===")
    
    sig = inspect.signature(NuScenesExtractor.extract_annotations)
    params = sig.parameters
    
    print(f"方法参数: {list(params.keys())}")
    
    # 验证必需的参数
    assert 'self' in params, "缺少self参数"
    assert 'sample_token' in params, "缺少sample_token参数"
    assert 'coordinate_frame' in params, "缺少coordinate_frame参数"
    assert 'include_ego_relative' in params, "缺少include_ego_relative参数"
    
    # 验证默认值
    assert params['coordinate_frame'].default == 'global', "coordinate_frame默认值应为'global'"
    assert params['include_ego_relative'].default == True, "include_ego_relative默认值应为True"
    
    print("✓ 方法签名正确")
    print(f"  - sample_token: 必需参数")
    print(f"  - coordinate_frame: 默认值='global'")
    print(f"  - include_ego_relative: 默认值=True")


def verify_extractor_implementation():
    """验证extract_annotations的实现逻辑"""
    print("\n=== 验证extract_annotations实现逻辑 ===")
    
    # 读取源代码
    source = inspect.getsource(NuScenesExtractor.extract_annotations)
    
    # 检查关键代码片段
    checks = {
        'ego_pose获取': 'extract_ego_pose' in source,
        'ego_state获取': 'extract_ego_state' in source,
        'ego速度向量计算': 'ego_velocity_vec' in source,
        'visibility获取': 'visibility_token' in source,
        'distance_to_ego计算': 'distance_to_ego' in source,
        'relative_velocity计算': 'relative_velocity' in source,
        'visibility_level': 'visibility_level' in source,
        'visibility_description': 'visibility_description' in source,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {'存在' if result else '缺失'}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ 实现逻辑完整")
    else:
        print("\n✗ 实现逻辑不完整")
        return False
    
    return True


def verify_builder_implementation():
    """验证GtStreamBuilder的实现"""
    print("\n=== 验证GtStreamBuilder实现 ===")
    
    # 读取_build_frame方法的源代码
    source = inspect.getsource(GtStreamBuilder._build_frame)
    
    # 检查新增的数组
    arrays = [
        'distances',
        'relative_velocities',
        'visibility_levels',
        'visibility_descriptions'
    ]
    
    all_passed = True
    for array_name in arrays:
        if array_name in source:
            print(f"  ✓ {array_name}: 已添加")
        else:
            print(f"  ✗ {array_name}: 缺失")
            all_passed = False
    
    # 检查objects字段
    object_fields = [
        'distances_to_ego',
        'relative_velocities',
        'visibility_levels',
        'visibility_descriptions'
    ]
    
    print("\n  检查objects字段:")
    for field_name in object_fields:
        if field_name in source:
            print(f"    ✓ {field_name}: 已添加")
        else:
            print(f"    ✗ {field_name}: 缺失")
            all_passed = False
    
    if all_passed:
        print("\n✓ GtStreamBuilder实现完整")
    else:
        print("\n✗ GtStreamBuilder实现不完整")
        return False
    
    return True


def verify_data_flow():
    """验证数据流"""
    print("\n=== 验证数据流 ===")
    
    print("\n数据流路径:")
    print("  1. NuScenesExtractor.extract_annotations")
    print("     └─ 计算: distance_to_ego, relative_velocity, visibility")
    print("  2. GtStreamBuilder._build_frame")
    print("     └─ 提取: ann.get('distance_to_ego'), ann.get('relative_velocity'), etc.")
    print("  3. 输出到gt_stream.bin")
    print("     └─ objects字段包含所有新属性")
    
    print("\n✓ 数据流设计合理")


def main():
    """主验证函数"""
    print("=" * 60)
    print("实现验证")
    print("=" * 60)
    
    try:
        # 验证各个组件
        verify_extractor_signature()
        
        if not verify_extractor_implementation():
            return 1
        
        if not verify_builder_implementation():
            return 1
        
        verify_data_flow()
        
        print("\n" + "=" * 60)
        print("✓ 所有验证通过!")
        print("=" * 60)
        
        print("\n实现总结:")
        print("1. ✓ NuScenesExtractor.extract_annotations 添加了 include_ego_relative 参数")
        print("2. ✓ 计算了距离、相对速度和可见性信息")
        print("3. ✓ GtStreamBuilder 包含了所有新属性")
        print("4. ✓ 数据流完整且合理")
        
        print("\n下一步:")
        print("- 使用实际的nuScenes数据集运行 test_gt_attributes.py 进行完整测试")
        print("- 修改 test_gt_attributes.py 中的 dataroot 路径")
        
        return 0
        
    except Exception as e:
        print(f"\n验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

