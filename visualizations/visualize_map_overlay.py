import msgpack
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_static_map(file_path):
    """加载静态地图数据"""
    with open(file_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return data

def load_prediction_stream(file_path):
    """加载预测流数据"""
    with open(file_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return data

def visualize_map_overlay(static_map_path, prediction_stream_path, frame_idx=0, output_path=None):
    """
    将预测地图与静态地图叠加显示
    
    Args:
        static_map_path: 静态地图文件路径
        prediction_stream_path: 预测流文件路径
        frame_idx: 要可视化的帧索引
        output_path: 可选，保存图片的路径
    """
    # 加载静态地图
    static_map = load_static_map(static_map_path)
    
    # 加载预测流
    prediction_stream = load_prediction_stream(prediction_stream_path)
    frames = prediction_stream.get('frames', [])
    
    if frame_idx >= len(frames):
        print(f"警告: 帧索引 {frame_idx} 超出范围，使用第0帧")
        frame_idx = 0
    
    frame = frames[frame_idx]
    predicted_mapping = frame.get('mapping', {})
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 绘制静态地图
    static_dividers = static_map.get('divider', [])
    static_boundaries = static_map.get('boundary', [])
    static_ped_crossings = static_map.get('ped_crossing', [])
    
    divider_labeled = False
    boundary_labeled = False
    crossing_labeled = False
    
    # 绘制静态地图的分隔线（蓝色，实线，较粗）
    for divider in static_dividers:
        if divider:
            coords = np.array(divider)
            ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, alpha=0.7, 
                   label='Static Divider' if not divider_labeled else '')
            divider_labeled = True
    
    # 绘制静态地图的边界（红色，实线，较粗）
    for boundary in static_boundaries:
        if boundary:
            coords = np.array(boundary)
            ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2.5, alpha=0.7,
                   label='Static Boundary' if not boundary_labeled else '')
            boundary_labeled = True
    
    # 绘制静态地图的人行横道（黄色，实线，较粗）
    for crossing in static_ped_crossings:
        if crossing:
            coords = np.array(crossing)
            ax.plot(coords[:, 0], coords[:, 1], 'y-', linewidth=2, alpha=0.6,
                   label='Static Ped Crossing' if not crossing_labeled else '')
            crossing_labeled = True
    
    # 绘制预测地图
    predicted_dividers = predicted_mapping.get('dividers', [])
    predicted_boundaries = predicted_mapping.get('boundaries', [])
    predicted_ped_crossings = predicted_mapping.get('ped_crossings', [])
    
    pred_divider_labeled = False
    pred_boundary_labeled = False
    pred_crossing_labeled = False
    
    # 绘制预测地图的分隔线（青色，虚线，较细）
    for divider in predicted_dividers:
        if divider:
            coords = np.array(divider)
            ax.plot(coords[:, 0], coords[:, 1], 'c--', linewidth=1.5, alpha=0.8,
                   label='Predicted Divider' if not pred_divider_labeled else '')
            pred_divider_labeled = True
    
    # 绘制预测地图的边界（品红色，虚线，较细）
    for boundary in predicted_boundaries:
        if boundary:
            coords = np.array(boundary)
            ax.plot(coords[:, 0], coords[:, 1], 'm--', linewidth=2, alpha=0.8,
                   label='Predicted Boundary' if not pred_boundary_labeled else '')
            pred_boundary_labeled = True
    
    # 绘制预测地图的人行横道（橙色，虚线，较细）
    for crossing in predicted_ped_crossings:
        if crossing:
            coords = np.array(crossing)
            ax.plot(coords[:, 0], coords[:, 1], 'orange', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Predicted Ped Crossing' if not pred_crossing_labeled else '')
            pred_crossing_labeled = True
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Map Overlay Visualization - Frame {frame_idx}\n'
                f'Static Map (solid) vs Predicted Map (dashed)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='将预测地图与静态地图叠加可视化')
    parser.add_argument('--static_map', type=str, 
                       default='../output/scene-0103/static_map.bin',
                       help='静态地图文件路径')
    parser.add_argument('--prediction_stream', type=str,
                       default='../output/scene-0103/prediction_stream.bin',
                       help='预测流文件路径')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='要可视化的帧索引（默认: 0）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，如果不指定则显示）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.static_map):
        print(f"错误: 静态地图文件不存在: {args.static_map}")
        return
    
    if not os.path.exists(args.prediction_stream):
        print(f"错误: 预测流文件不存在: {args.prediction_stream}")
        return
    
    visualize_map_overlay(
        args.static_map,
        args.prediction_stream,
        args.frame_idx,
        args.output
    )

if __name__ == '__main__':
    main()


