"""
测试GT对象新属性的正确性

验证：
1. 距离计算的准确性
2. 相对速度的方向和大小
3. 可见性信息的完整性
4. 边界情况处理
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data_converter.core.nuscenes_extractor import NuScenesExtractor


def test_distance_calculation(extractor, sample_token):
    """测试距离计算"""
    print("\n=== 测试距离计算 ===")
    
    # 获取ego pose和annotations
    ego_pose = extractor.extract_ego_pose(sample_token)
    annotations = extractor.extract_annotations(sample_token, include_ego_relative=True)
    
    ego_translation = np.array(ego_pose['translation'])
    
    for ann in annotations[:3]:  # 只测试前3个对象
        obj_translation = np.array(ann['translation'])
        
        # 手动计算距离
        manual_distance = np.linalg.norm(obj_translation[:2] - ego_translation[:2])
        
        # 从annotation获取的距离
        extracted_distance = ann.get('distance_to_ego', None)
        
        print(f"\n对象类别: {ann['category']}")
        print(f"  手动计算距离: {manual_distance:.2f}m")
        print(f"  提取的距离: {extracted_distance:.2f}m" if extracted_distance is not None else "  提取的距离: None")
        
        if extracted_distance is not None:
            diff = abs(manual_distance - extracted_distance)
            print(f"  差异: {diff:.6f}m")
            assert diff < 0.001, f"距离计算误差过大: {diff}"
    
    print("\n✓ 距离计算测试通过")


def test_relative_velocity(extractor, sample_token):
    """测试相对速度计算"""
    print("\n=== 测试相对速度计算 ===")
    
    # 获取ego信息
    ego_pose = extractor.extract_ego_pose(sample_token)
    ego_state = extractor.extract_ego_state(sample_token)
    annotations = extractor.extract_annotations(sample_token, include_ego_relative=True)
    
    # 计算ego速度向量
    ego_velocity = ego_state['velocity']
    ego_yaw = ego_pose['yaw']
    ego_velocity_vec = np.array([
        ego_velocity * np.cos(ego_yaw),
        ego_velocity * np.sin(ego_yaw)
    ])
    
    print(f"\nEgo速度: {ego_velocity:.2f} m/s")
    print(f"Ego yaw: {ego_yaw:.2f} rad")
    print(f"Ego速度向量: [{ego_velocity_vec[0]:.2f}, {ego_velocity_vec[1]:.2f}]")
    
    for ann in annotations[:3]:  # 只测试前3个对象
        obj_velocity = np.array(ann['velocity'])
        
        # 手动计算相对速度
        manual_rel_velocity = obj_velocity - ego_velocity_vec
        
        # 从annotation获取的相对速度
        extracted_rel_velocity = ann.get('relative_velocity', None)
        
        print(f"\n对象类别: {ann['category']}")
        print(f"  对象速度: [{obj_velocity[0]:.2f}, {obj_velocity[1]:.2f}]")
        print(f"  手动计算相对速度: [{manual_rel_velocity[0]:.2f}, {manual_rel_velocity[1]:.2f}]")
        
        if extracted_rel_velocity is not None:
            extracted_rel_velocity = np.array(extracted_rel_velocity)
            print(f"  提取的相对速度: [{extracted_rel_velocity[0]:.2f}, {extracted_rel_velocity[1]:.2f}]")
            
            diff = np.linalg.norm(manual_rel_velocity - extracted_rel_velocity)
            print(f"  差异: {diff:.6f}")
            assert diff < 0.001, f"相对速度计算误差过大: {diff}"
    
    print("\n✓ 相对速度计算测试通过")


def test_visibility_info(extractor, sample_token):
    """测试可见性信息"""
    print("\n=== 测试可见性信息 ===")
    
    annotations = extractor.extract_annotations(sample_token, include_ego_relative=True)
    
    visibility_stats = {
        0: 0,  # unknown
        1: 0,  # v80-100
        2: 0,  # v60-80
        3: 0,  # v40-60
        4: 0   # v0-40
    }
    
    for ann in annotations:
        visibility_level = ann.get('visibility_level', 0)
        visibility_desc = ann.get('visibility_description', 'unknown')
        
        visibility_stats[visibility_level] += 1
        
        # 验证level和description的一致性
        if visibility_level > 0:
            assert visibility_desc != 'unknown', f"可见性level={visibility_level}但description是unknown"
    
    print(f"\n可见性统计 (共{len(annotations)}个对象):")
    print(f"  Level 0 (未知): {visibility_stats[0]}")
    print(f"  Level 1 (v80-100): {visibility_stats[1]}")
    print(f"  Level 2 (v60-80): {visibility_stats[2]}")
    print(f"  Level 3 (v40-60): {visibility_stats[3]}")
    print(f"  Level 4 (v0-40): {visibility_stats[4]}")
    
    # 显示一些示例
    print("\n示例对象:")
    for ann in annotations[:5]:
        print(f"  {ann['category']}: level={ann.get('visibility_level', 0)}, "
              f"desc={ann.get('visibility_description', 'unknown')}")
    
    print("\n✓ 可见性信息测试通过")


def test_backward_compatibility(extractor, sample_token):
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    # 测试不包含ego相对信息的情况
    annotations_without = extractor.extract_annotations(sample_token, include_ego_relative=False)
    
    print(f"\n不包含ego相对信息时的annotation字段:")
    if annotations_without:
        ann = annotations_without[0]
        print(f"  包含的字段: {list(ann.keys())}")
        
        # 验证不包含新字段
        assert 'distance_to_ego' not in ann, "include_ego_relative=False时不应包含distance_to_ego"
        assert 'relative_velocity' not in ann, "include_ego_relative=False时不应包含relative_velocity"
        assert 'visibility_level' not in ann, "include_ego_relative=False时不应包含visibility_level"
    
    # 测试包含ego相对信息的情况
    annotations_with = extractor.extract_annotations(sample_token, include_ego_relative=True)
    
    print(f"\n包含ego相对信息时的annotation字段:")
    if annotations_with:
        ann = annotations_with[0]
        print(f"  包含的字段: {list(ann.keys())}")
        
        # 验证包含新字段
        assert 'distance_to_ego' in ann, "include_ego_relative=True时应包含distance_to_ego"
        assert 'relative_velocity' in ann, "include_ego_relative=True时应包含relative_velocity"
        assert 'visibility_level' in ann, "include_ego_relative=True时应包含visibility_level"
        assert 'visibility_description' in ann, "include_ego_relative=True时应包含visibility_description"
    
    print("\n✓ 向后兼容性测试通过")


def main():
    """主测试函数"""
    print("=" * 60)
    print("GT对象新属性测试")
    print("=" * 60)
    
    # 初始化extractor（需要提供实际的数据路径）
    # 注意：这里需要用户提供实际的nuScenes数据路径
    dataroot = "/path/to/nuscenes"  # 需要修改为实际路径
    
    try:
        print(f"\n初始化NuScenesExtractor...")
        print(f"数据路径: {dataroot}")
        
        extractor = NuScenesExtractor(dataroot=dataroot, version='v1.0-mini')
        
        # 获取第一个场景的第一个sample
        scene = extractor.nusc.scene[0]
        sample_token = scene['first_sample_token']
        
        print(f"测试场景: {scene['name']}")
        print(f"Sample token: {sample_token}")
        
        # 运行所有测试
        test_distance_calculation(extractor, sample_token)
        test_relative_velocity(extractor, sample_token)
        test_visibility_info(extractor, sample_token)
        test_backward_compatibility(extractor, sample_token)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n错误: 找不到nuScenes数据集")
        print(f"请修改脚本中的dataroot路径为实际的nuScenes数据集路径")
        print(f"当前路径: {dataroot}")
        return 1
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

