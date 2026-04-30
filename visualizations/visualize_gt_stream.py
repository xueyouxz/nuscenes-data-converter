import msgpack
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_gt_stream(file_path):
    """加载GT流数据"""
    with open(file_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return data

def visualize_gt_stream(gt_stream_path, frame_idx=0, output_path=None):
    """
    可视化GT流数据
    
    Args:
        gt_stream_path: GT流文件路径
        frame_idx: 要可视化的帧索引
        output_path: 可选，保存图片的路径
    """
    data = load_gt_stream(gt_stream_path)
    
    frames = data.get('frames', [])
    if frame_idx >= len(frames):
        print(f"警告: 帧索引 {frame_idx} 超出范围，使用第0帧")
        frame_idx = 0
    
    frame = frames[frame_idx]
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    ego_pose = frame.get('ego_pose', {})
    ego_translation = ego_pose.get('translation', [0, 0, 0])
    ax.plot(ego_translation[0], ego_translation[1], 'ro', markersize=12, label='Ego Vehicle')
    
    objects = frame.get('objects', {})
    instance_tokens = objects.get('instance_tokens', [])
    categories = objects.get('categories', [])
    boxes = objects.get('boxes', [])
    velocities = objects.get('velocities', [])
    
    class_colors = {
        'car': 'blue',
        'truck': 'cyan',
        'bus': 'navy',
        'pedestrian': 'red',
        'bicycle': 'green',
        'motorcycle': 'lime',
        'barrier': 'gray',
        'traffic_cone': 'yellow',
    }
    
    seen_categories = set()
    for i, box in enumerate(boxes):
        if i < len(categories):
            x, y, z, w, l, h, yaw = box[:7]
            category = categories[i]
            
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            corners = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2],
                [-l/2, -w/2]
            ])
            
            rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners_rotated = corners @ rot_matrix.T
            corners_rotated[:, 0] += x
            corners_rotated[:, 1] += y
            
            color = class_colors.get(category, 'gray')
            label = category if category not in seen_categories else ''
            seen_categories.add(category)
            ax.plot(corners_rotated[:, 0], corners_rotated[:, 1], color=color, linewidth=1.5, alpha=0.7, label=label)
            
            if i < len(velocities) and velocities[i]:
                vx, vy = velocities[i][:2]
                if vx != 0 or vy != 0:
                    ax.arrow(x, y, vx * 0.5, vy * 0.5, head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=0.6)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'GT Stream Visualization - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='可视化GT流数据')
    parser.add_argument('--gt_stream', type=str,
                       default='../output/scene-0103/gt_stream.bin',
                       help='GT流文件路径')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='要可视化的帧索引（默认: 0）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，如果不指定则显示）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_stream):
        print(f"错误: GT流文件不存在: {args.gt_stream}")
        return
    
    visualize_gt_stream(args.gt_stream, args.frame_idx, args.output)

if __name__ == '__main__':
    main()

