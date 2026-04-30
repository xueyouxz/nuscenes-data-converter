import msgpack
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_prediction_stream(file_path):
    """加载预测流数据"""
    with open(file_path, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    return data

def visualize_prediction_stream(prediction_stream_path, frame_idx=0, output_path=None):
    """
    可视化预测流数据
    
    Args:
        prediction_stream_path: 预测流文件路径
        frame_idx: 要可视化的帧索引
        output_path: 可选，保存图片的路径
    """
    data = load_prediction_stream(prediction_stream_path)
    
    frames = data.get('frames', [])
    if frame_idx >= len(frames):
        print(f"警告: 帧索引 {frame_idx} 超出范围，使用第0帧")
        frame_idx = 0
    
    frame = frames[frame_idx]
    
    fig, ax = plt.subplots(figsize=(14, 14))

    detection = frame.get('detection', {})
    boxes = detection.get('boxes', [])
    scores = detection.get('scores', [])
    classes = detection.get('classes', [])
    trajectories = detection.get('trajectories', [])
    
    for i, box in enumerate(boxes):
        if i < len(scores) and scores[i] > 0.3:
            x, y, z, w, l, h, yaw = box[:7]
            
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
            
            color = plt.cm.tab10(i % 10)
            ax.plot(corners_rotated[:, 0], corners_rotated[:, 1], color=color, linewidth=1.5, alpha=0.7)
            
            if i < len(trajectories) and trajectories[i]:
                traj = np.array(trajectories[i])
                if len(traj.shape) == 3 and traj.shape[0] > 0:
                    traj = traj[0]
                if len(traj.shape) == 2 and traj.shape[0] > 0:
                    ax.plot(traj[:, 0], traj[:, 1], '--', color=color, linewidth=1, alpha=0.5)
    
    planning = frame.get('planning', {})
    planning_traj = planning.get('trajectory', []) if isinstance(planning, dict) else planning
    if planning_traj:
        planning_traj = np.array(planning_traj)
        if len(planning_traj.shape) == 2:
            ax.plot(planning_traj[:, 0], planning_traj[:, 1], 'g-', linewidth=3, alpha=0.8, label='Planning Trajectory')
    
    mapping = frame.get('mapping', {})
    dividers = mapping.get('dividers', [])
    boundaries = mapping.get('boundaries', [])
    ped_crossings = mapping.get('ped_crossings', [])
    
    for divider in dividers:
        if divider:
            coords = np.array(divider)
            ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1, alpha=0.6)
    
    for boundary in boundaries:
        if boundary:
            coords = np.array(boundary)
            ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=1.5, alpha=0.6)
    
    for crossing in ped_crossings:
        if crossing:
            coords = np.array(crossing)
            ax.plot(coords[:, 0], coords[:, 1], 'y-', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Prediction Stream Visualization - Frame {frame_idx}', fontsize=14, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='可视化预测流数据')
    parser.add_argument('--prediction_stream', type=str,
                       default='../output/scene-0103/prediction_stream.bin',
                       help='预测流文件路径')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='要可视化的帧索引（默认: 0）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，如果不指定则显示）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prediction_stream):
        print(f"错误: 预测流文件不存在: {args.prediction_stream}")
        return
    
    visualize_prediction_stream(args.prediction_stream, args.frame_idx, args.output)

if __name__ == '__main__':
    main()

