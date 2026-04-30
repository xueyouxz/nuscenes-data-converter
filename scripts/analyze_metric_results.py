#!/usr/bin/env python3
"""
分析SparseDrive模型输出中的metric_results字段

该脚本用于详细分析pkl文件中的metric_results部分
"""

import pickle
import torch
import json
import sys


def load_prediction_data(file_path):
    """强制CPU加载pkl文件"""
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    
    original_restore = torch.serialization.default_restore_location
    torch.serialization.default_restore_location = lambda storage, location: storage
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    finally:
        torch.cuda.is_available = original_is_available
        torch.serialization.default_restore_location = original_restore


def analyze_metric_results(predictions):
    """分析metric_results的结构"""
    print("="*60)
    print("Metric Results 分析")
    print("="*60)
    
    # 检查第一个样本
    sample = predictions[0]
    
    if 'metric_results' not in sample:
        print("未找到 metric_results 字段")
        return None
    
    metric_results = sample['metric_results']
    
    print(f"\n数据类型: {type(metric_results)}")
    
    if isinstance(metric_results, dict):
        print(f"字段数量: {len(metric_results)}")
        print(f"\n字段列表:")
        
        analysis = {}
        for key, value in metric_results.items():
            print(f"\n  {key}:")
            print(f"    类型: {type(value)}")
            
            field_info = {"type": str(type(value).__name__)}
            
            if isinstance(value, (int, float)):
                print(f"    值: {value}")
                field_info["value"] = value
            elif isinstance(value, str):
                print(f"    值: {value[:100]}")
                field_info["value"] = value[:100]
            elif isinstance(value, dict):
                print(f"    子字段: {list(value.keys())}")
                field_info["keys"] = list(value.keys())
                
                # 递归分析子字段
                sub_analysis = {}
                for sub_key, sub_value in value.items():
                    sub_type = type(sub_value).__name__
                    if isinstance(sub_value, (list, tuple)):
                        sub_analysis[sub_key] = {
                            "type": sub_type,
                            "length": len(sub_value),
                            "sample": str(sub_value)[:100]
                        }
                    elif isinstance(sub_value, dict):
                        sub_analysis[sub_key] = {
                            "type": sub_type,
                            "keys": list(sub_value.keys())
                        }
                    else:
                        sub_analysis[sub_key] = {
                            "type": sub_type,
                            "value": str(sub_value)[:100]
                        }
                field_info["sub_fields"] = sub_analysis
            elif isinstance(value, (list, tuple)):
                print(f"    长度: {len(value)}")
                if len(value) > 0:
                    print(f"    元素类型: {type(value[0])}")
                    print(f"    示例: {str(value[0])[:100]}")
                field_info.update({
                    "length": len(value),
                    "element_type": type(value[0]).__name__ if len(value) > 0 else "empty"
                })
            
            analysis[key] = field_info
        
        return analysis
    else:
        print(f"值: {metric_results}")
        return None


def summarize_all_metrics(predictions):
    """统计所有样本的metric_results"""
    print("\n" + "="*60)
    print("所有样本的Metric统计")
    print("="*60)
    
    all_metrics = {}
    
    for i, sample in enumerate(predictions):
        if 'metric_results' not in sample:
            continue
        
        metric_results = sample['metric_results']
        
        if isinstance(metric_results, dict):
            for key, value in metric_results.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                
                # 收集数值型指标
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
                elif isinstance(value, dict):
                    # 对于嵌套字典，收集所有数值
                    for sub_key, sub_value in value.items():
                        metric_name = f"{key}.{sub_key}"
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        if isinstance(sub_value, (int, float)):
                            all_metrics[metric_name].append(sub_value)
    
    # 计算统计信息
    print("\n数值型指标统计:")
    statistics = {}
    for key, values in all_metrics.items():
        if len(values) > 0 and all(isinstance(v, (int, float)) for v in values):
            import numpy as np
            values_array = np.array(values)
            stats = {
                "count": len(values),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
            }
            statistics[key] = stats
            
            print(f"\n  {key}:")
            print(f"    样本数: {stats['count']}")
            print(f"    均值: {stats['mean']:.4f}")
            print(f"    标准差: {stats['std']:.4f}")
            print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return statistics


def main():
    pkl_file = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl"
    output_file = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/scripts/metric_results_analysis.json"
    
    print("正在加载pkl文件...")
    predictions = load_prediction_data(pkl_file)
    print(f"✓ 成功加载，共 {len(predictions)} 个样本\n")
    
    # 分析第一个样本的metric_results结构
    analysis = analyze_metric_results(predictions)
    
    # 统计所有样本的metrics
    statistics = summarize_all_metrics(predictions)
    
    # 保存结果
    if analysis is not None or statistics is not None:
        result = {
            "sample_structure": analysis,
            "statistics": statistics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 分析结果已保存到: {output_file}")
    
    print("\n✓ 分析完成！")


if __name__ == "__main__":
    main()



