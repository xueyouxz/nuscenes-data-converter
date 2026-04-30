#!/usr/bin/env python3
"""
SparseDrive模型输出PKL文件分析脚本

该脚本用于分析SparseDrive模型输出的pkl文件，提取数据结构、字段信息和示例数据。
"""

import pickle
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import sys


def load_prediction_data(file_path: str) -> List[Dict]:
    """
    强制CPU加载pkl文件，避免CUDA依赖问题
    
    Args:
        file_path: pkl文件路径
    
    Returns:
        预测数据列表
    """
    # 临时禁用CUDA
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    
    # 临时修改设备恢复逻辑
    original_restore = torch.serialization.default_restore_location
    torch.serialization.default_restore_location = lambda storage, location: storage
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    finally:
        # 恢复原始函数
        torch.cuda.is_available = original_is_available
        torch.serialization.default_restore_location = original_restore


def analyze_tensor_or_array(data, name: str) -> Dict[str, Any]:
    """
    分析Tensor或数组的属性
    
    Args:
        data: Tensor或numpy数组
        name: 字段名称
    
    Returns:
        包含属性信息的字典
    """
    info = {
        "name": name,
        "type": str(type(data).__name__)
    }
    
    # 处理Tensor
    if torch.is_tensor(data):
        info.update({
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "requires_grad": data.requires_grad,
        })
        if data.numel() > 0:
            info["min"] = float(data.min().item())
            info["max"] = float(data.max().item())
            # 对于整数类型，转换为float后再计算mean
            if data.dtype in [torch.int32, torch.int64, torch.long]:
                info["mean"] = float(data.float().mean().item())
            elif data.dtype in [torch.float32, torch.float64, torch.float16]:
                info["mean"] = float(data.mean().item())
        else:
            info["min"] = None
            info["max"] = None
            info["mean"] = None
    # 处理numpy数组
    elif isinstance(data, np.ndarray):
        info.update({
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        })
        if data.size > 0:
            info["min"] = float(data.min())
            info["max"] = float(data.max())
            # 对于整数类型，转换为float后再计算mean
            if np.issubdtype(data.dtype, np.integer):
                info["mean"] = float(data.astype(np.float64).mean())
            else:
                info["mean"] = float(data.mean())
        else:
            info["min"] = None
            info["max"] = None
            info["mean"] = None
    # 处理列表
    elif isinstance(data, list):
        info.update({
            "length": len(data),
            "element_type": str(type(data[0]).__name__) if len(data) > 0 else "empty"
        })
        if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
            info["element_shape"] = list(np.array(data[0]).shape) if len(data) > 0 else []
    
    return info


def analyze_sample_structure(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析单个样本的结构
    
    Args:
        sample: 样本字典
    
    Returns:
        样本结构分析结果
    """
    structure = {}
    
    for key, value in sample.items():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            structure[key] = analyze_tensor_or_array(value, key)
        elif isinstance(value, list):
            structure[key] = analyze_tensor_or_array(value, key)
        elif isinstance(value, dict):
            structure[key] = {
                "type": "dict",
                "keys": list(value.keys())
            }
        else:
            structure[key] = {
                "type": str(type(value).__name__),
                "value": str(value)[:100]  # 截断长字符串
            }
    
    return structure


def convert_to_json_serializable(obj):
    """
    将对象转换为JSON可序列化的格式
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def extract_sample_data(sample: Dict[str, Any], max_points: int = 5) -> Dict[str, Any]:
    """
    提取样本数据示例（取前几个点）
    
    Args:
        sample: 样本字典
        max_points: 最大提取点数
    
    Returns:
        样本数据示例
    """
    example = {}
    
    for key, value in sample.items():
        try:
            if torch.is_tensor(value):
                value_np = value.detach().cpu().numpy()
                if value_np.size > 0:
                    # 根据维度提取示例
                    if value_np.ndim == 1:
                        example[key] = value_np[:min(max_points, len(value_np))].tolist()
                    elif value_np.ndim == 2:
                        example[key] = value_np[:min(max_points, len(value_np))].tolist()
                    elif value_np.ndim >= 3:
                        example[key] = value_np[:min(2, len(value_np))].tolist()
                else:
                    example[key] = []
            elif isinstance(value, np.ndarray):
                if value.size > 0:
                    if value.ndim == 1:
                        example[key] = value[:min(max_points, len(value))].tolist()
                    elif value.ndim == 2:
                        example[key] = value[:min(max_points, len(value))].tolist()
                    elif value.ndim >= 3:
                        example[key] = value[:min(2, len(value))].tolist()
                else:
                    example[key] = []
            elif isinstance(value, list):
                if len(value) > 0:
                    # 递归转换列表中的元素
                    sampled = value[:min(max_points, len(value))]
                    example[key] = convert_to_json_serializable(sampled)
                else:
                    example[key] = []
            else:
                example[key] = str(value)[:200]  # 截断长字符串
        except Exception as e:
            example[key] = f"<提取失败: {str(e)[:50]}>"
    
    return example


def generate_field_documentation(structure: Dict[str, Any]) -> List[str]:
    """
    根据字段结构生成文档说明
    
    Args:
        structure: 字段结构信息
    
    Returns:
        文档行列表
    """
    docs = []
    
    # 根据sparsedrive_service.py中的解析逻辑添加字段说明
    field_descriptions = {
        "token": "样本的唯一标识符（sample token）",
        "boxes_3d": "3D检测框，包含位置、尺寸、朝向等信息，shape: (N, 9)，其中N为检测到的目标数量。\n"
                   "        每个检测框包含9个值: [x, y, z, w, l, h, yaw, vx, vy]\n"
                   "        - x, y: 目标在ego坐标系下的横向和纵向位置（米）\n"
                   "        - z: 目标高度（米）\n"
                   "        - w, l, h: 目标的宽度、长度、高度（米）\n"
                   "        - yaw: 目标的朝向角（弧度）\n"
                   "        - vx, vy: 目标的速度（米/秒）",
        "scores_3d": "3D检测框的置信度分数，shape: (N,)，值域[0, 1]",
        "labels_3d": "3D检测框的类别标签，shape: (N,)，对应CLASSES中的索引\n"
                    "        类别包括: car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier",
        "instance_ids": "目标实例ID，用于跨帧跟踪，shape: (N,)",
        "trajs_3d": "多模态轨迹预测，shape: (N, 6, 12, 2)\n"
                   "        - N: 目标数量\n"
                   "        - 6: 每个目标预测6个模态轨迹\n"
                   "        - 12: 每个轨迹预测未来12个时间步（通常每步0.5秒，共6秒）\n"
                   "        - 2: 每个点的2D坐标 (x, y)",
        "trajs_score": "轨迹置信度分数，shape: (N, 6)，每个目标的6个模态轨迹各有一个分数",
        "vectors": "地图矢量预测，列表形式，每个元素是一个地图元素的点序列\n"
                  "        shape: List[(M, 2)]，其中M为每个地图元素的点数",
        "scores": "地图元素的置信度分数",
        "labels": "地图元素的类别标签，对应MAP_CLASSES中的索引\n"
                 "        类别包括: ped_crossing（人行横道）, divider（分隔线）, boundary（边界）",
        "planning": "多模态规划轨迹，shape: (3, 6, 12, 2)\n"
                   "        - 3: 三种驾驶指令（左转、直行、右转）\n"
                   "        - 6: 每个指令下预测6个模态轨迹\n"
                   "        - 12: 每个轨迹预测未来12个时间步\n"
                   "        - 2: 每个点的2D坐标 (x, y)",
        "planning_score": "规划轨迹的置信度分数，shape: (3, 6)",
        "final_planning": "最终选定的规划轨迹，shape: (12, 2)，表示未来12个时间步的规划路径",
    }
    
    for field_name, field_info in structure.items():
        docs.append(f"\n### {field_name}")
        
        # 添加字段说明
        if field_name in field_descriptions:
            docs.append(f"**说明**: {field_descriptions[field_name]}")
        
        # 添加类型和形状信息
        if "shape" in field_info:
            docs.append(f"- **类型**: {field_info['type']}")
            docs.append(f"- **形状**: {field_info['shape']}")
            if "dtype" in field_info:
                docs.append(f"- **数据类型**: {field_info['dtype']}")
            if "min" in field_info and field_info["min"] is not None:
                docs.append(f"- **数值范围**: [{field_info['min']:.4f}, {field_info['max']:.4f}]")
                docs.append(f"- **平均值**: {field_info['mean']:.4f}")
        elif "length" in field_info:
            docs.append(f"- **类型**: {field_info['type']}")
            docs.append(f"- **长度**: {field_info['length']}")
            if "element_type" in field_info:
                docs.append(f"- **元素类型**: {field_info['element_type']}")
            if "element_shape" in field_info:
                docs.append(f"- **元素形状**: {field_info['element_shape']}")
        else:
            docs.append(f"- **类型**: {field_info['type']}")
            if "value" in field_info:
                docs.append(f"- **值**: {field_info['value']}")
    
    return docs


def main():
    """主函数"""
    # 文件路径
    pkl_file = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl"
    output_json = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/scripts/sparsedrive_analysis.json"
    output_doc = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/docs/sparsedrive_data_structure.md"
    
    print(f"正在加载pkl文件: {pkl_file}")
    try:
        prediction_data = load_prediction_data(pkl_file)
        print(f"✓ 成功加载，共 {len(prediction_data)} 个样本")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        sys.exit(1)
    
    # 分析数据结构
    print("\n分析数据结构...")
    
    # 获取顶层结构
    print(f"顶层数据类型: {type(prediction_data)}")
    print(f"样本数量: {len(prediction_data)}")
    
    if len(prediction_data) > 0:
        first_sample = prediction_data[0]
        print(f"单个样本类型: {type(first_sample)}")
        
        if isinstance(first_sample, dict):
            print(f"样本顶层keys: {list(first_sample.keys())}")
            
            # 检查是否有嵌套的'img_bbox'结构
            if 'img_bbox' in first_sample:
                print(f"'img_bbox'结构的keys: {list(first_sample['img_bbox'].keys())}")
                analysis_target = first_sample['img_bbox']
            else:
                analysis_target = first_sample
            
            # 分析结构
            structure = analyze_sample_structure(analysis_target)
            
            # 提取示例数据
            example_data = extract_sample_data(analysis_target)
            
            # 保存分析结果到JSON
            analysis_result = {
                "总样本数": len(prediction_data),
                "数据结构": structure,
                "示例数据": example_data,
                "顶层keys": list(first_sample.keys()),
            }
            
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 分析结果已保存到: {output_json}")
            
            # 生成Markdown文档
            print("\n生成文档...")
            doc_lines = []
            
            # 文档标题和介绍
            doc_lines.extend([
                "# SparseDrive模型输出数据结构文档",
                "",
                "## 概述",
                "",
                "SparseDrive是一个端到端的自动驾驶感知和规划模型，能够同时预测：",
                "- **3D目标检测**: 检测周围车辆、行人等目标",
                "- **轨迹预测**: 预测检测目标的未来运动轨迹（多模态）",
                "- **地图感知**: 预测道路地图元素（车道线、人行横道等）",
                "- **运动规划**: 生成自车的规划轨迹（多模态）",
                "",
                f"## 数据文件信息",
                "",
                f"- **文件路径**: `{pkl_file}`",
                f"- **文件格式**: Python Pickle (*.pkl)",
                f"- **总样本数**: {len(prediction_data)}",
                f"- **数据结构**: 列表，每个元素对应一个时间帧的预测结果",
                "",
                "## 顶层数据结构",
                "",
                "```python",
                f"数据类型: {type(prediction_data).__name__}",
                f"样本数量: {len(prediction_data)}",
                "```",
                "",
                "每个样本是一个字典，包含以下顶层keys:",
                "```python",
                f"{list(first_sample.keys())}",
                "```",
                "",
            ])
            
            # 说明img_bbox结构
            if 'img_bbox' in first_sample:
                doc_lines.extend([
                    "实际的预测数据存储在 `img_bbox` 字段中。",
                    "",
                    "## 预测数据字段详解 (img_bbox)",
                    "",
                ])
            else:
                doc_lines.extend([
                    "## 预测数据字段详解",
                    "",
                ])
            
            # 生成字段文档
            field_docs = generate_field_documentation(structure)
            doc_lines.extend(field_docs)
            
            # 添加使用示例
            doc_lines.extend([
                "",
                "## 数据加载示例",
                "",
                "### Python代码",
                "",
                "```python",
                "import pickle",
                "import torch",
                "",
                "def load_sparsedrive_predictions(file_path):",
                "    # 强制CPU加载，避免CUDA问题",
                "    original_is_available = torch.cuda.is_available",
                "    torch.cuda.is_available = lambda: False",
                "    ",
                "    original_restore = torch.serialization.default_restore_location",
                "    torch.serialization.default_restore_location = lambda storage, location: storage",
                "    ",
                "    try:",
                "        with open(file_path, 'rb') as f:",
                "            data = pickle.load(f)",
                "        return data",
                "    finally:",
                "        torch.cuda.is_available = original_is_available",
                "        torch.serialization.default_restore_location = original_restore",
                "",
                "# 加载数据",
                "predictions = load_sparsedrive_predictions('sparsedrive_stage2_trainval_with_metric.pkl')",
                "",
                "# 访问第一个样本",
                "sample = predictions[0]['img_bbox']",
                "sample_token = sample['token']",
                "boxes_3d = sample['boxes_3d']  # 3D检测框",
                "planning = sample['planning']  # 规划轨迹",
                "```",
                "",
                "## 坐标系统说明",
                "",
                "### Ego坐标系",
                "模型输出的所有坐标都是相对于自车（ego vehicle）的局部坐标系：",
                "- **X轴**: 指向车辆前方",
                "- **Y轴**: 指向车辆左侧",
                "- **Z轴**: 指向车辆上方",
                "- **原点**: 位于自车中心",
                "",
                "### 坐标转换",
                "如需将预测结果转换到全局坐标系，需要使用样本对应的ego_pose信息：",
                "```python",
                "from nuscenes.prediction.helper import convert_local_coords_to_global",
                "",
                "# 假设已获取ego_pose",
                "ego_translation = ego_pose['translation']",
                "ego_rotation = ego_pose['rotation']",
                "",
                "# 转换局部坐标到全局坐标",
                "global_coords = convert_local_coords_to_global(",
                "    local_coords, ego_translation, ego_rotation",
                ")",
                "```",
                "",
                "## 类别定义",
                "",
                "### 目标检测类别 (CLASSES)",
                "```python",
                "CLASSES = (",
                "    'car',                    # 0: 小汽车",
                "    'truck',                  # 1: 卡车",
                "    'trailer',                # 2: 拖车",
                "    'bus',                    # 3: 公交车",
                "    'construction_vehicle',   # 4: 工程车辆",
                "    'bicycle',                # 5: 自行车",
                "    'motorcycle',             # 6: 摩托车",
                "    'pedestrian',             # 7: 行人",
                "    'traffic_cone',           # 8: 交通锥",
                "    'barrier',                # 9: 护栏",
                ")",
                "```",
                "",
                "### 地图元素类别 (MAP_CLASSES)",
                "```python",
                "MAP_CLASSES = (",
                "    'ped_crossing',  # 0: 人行横道",
                "    'divider',       # 1: 车道分隔线",
                "    'boundary',      # 2: 道路边界",
                ")",
                "```",
                "",
                "## 数据处理流程",
                "",
                "参考 `temp/services/sparsedrive_service.py` 中的实现：",
                "",
                "1. **加载预测结果**: 使用`load_prediction_data()`方法",
                "2. **按token索引**: 将预测结果组织为 `sample_token -> prediction` 的映射",
                "3. **坐标转换**: 将ego坐标系转换为全局坐标系",
                "4. **阈值过滤**: 过滤置信度低于0.3的预测结果",
                "5. **多模态处理**: 根据驾驶指令选择对应的规划轨迹",
                "",
                "## 注意事项",
                "",
                "1. **设备兼容性**: pkl文件可能包含GPU tensor，加载时需要处理设备映射",
                "2. **内存占用**: 完整数据集较大，建议按需加载",
                "3. **坐标一致性**: 注意区分局部坐标系和全局坐标系",
                "4. **置信度阈值**: 建议使用0.3作为过滤阈值",
                "5. **轨迹时间步**: 每个时间步通常对应0.5秒",
                "",
                "## 参考资料",
                "",
                "- SparseDrive论文: [链接]",
                "- nuScenes数据集: https://www.nuscenes.org/",
                "- 代码仓库: temp/services/sparsedrive_service.py",
                "",
            ])
            
            # 保存文档
            with open(output_doc, 'w', encoding='utf-8') as f:
                f.write('\n'.join(doc_lines))
            print(f"✓ 文档已保存到: {output_doc}")
            
            # 打印摘要
            print("\n" + "="*60)
            print("数据结构摘要")
            print("="*60)
            for key, info in structure.items():
                if "shape" in info:
                    print(f"{key:20s} | shape: {info['shape']}")
                elif "length" in info:
                    print(f"{key:20s} | length: {info['length']}")
                else:
                    print(f"{key:20s} | type: {info['type']}")
            
            print("\n✓ 分析完成！")
    else:
        print("警告: 数据为空")


if __name__ == "__main__":
    main()

