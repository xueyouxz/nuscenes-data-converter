#!/usr/bin/env python3
"""
SparseDrive预测结果可视化示例脚本

该脚本展示如何加载和可视化SparseDrive模型的预测结果
包括3D检测框、轨迹预测、地图元素和规划轨迹的可视化
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import LineCollection
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


# 类别定义
CLASSES = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)

MAP_CLASSES = ('ped_crossing', 'divider', 'boundary')

# 颜色映射
CLASS_COLORS = {
    'car': 'blue',
    'truck': 'cyan',
    'trailer': 'lightblue',
    'bus': 'navy',
    'construction_vehicle': 'orange',
    'bicycle': 'green',
    'motorcycle': 'lime',
    'pedestrian': 'red',
    'traffic_cone': 'yellow',
    'barrier': 'gray',
}

MAP_COLORS = {
    'ped_crossing': 'yellow',
    'divider': 'white',
    'boundary': 'red',
}


def visualize_sample(sample_data, sample_idx=0, score_threshold=0.3):
    """
    可视化单个样本的预测结果
    
    Args:
        sample_data: 样本数据字典 (img_bbox)
        sample_idx: 样本索引
        score_threshold: 置信度阈值
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f'SparseDrive预测结果可视化 - Sample {sample_idx}\nToken: {sample_data["token"]}', 
                 fontsize=16, fontweight='bold')
    
    # 1. 3D检测框可视化
    ax1 = axes[0, 0]
    visualize_detections(ax1, sample_data, score_threshold)
    
    # 2. 轨迹预测可视化
    ax2 = axes[0, 1]
    visualize_trajectories(ax2, sample_data, score_threshold)
    
    # 3. 地图元素可视化
    ax3 = axes[1, 0]
    visualize_map(ax3, sample_data, score_threshold)
    
    # 4. 规划轨迹可视化
    ax4 = axes[1, 1]
    visualize_planning(ax4, sample_data)
    
    plt.tight_layout()
    return fig


def visualize_detections(ax, sample_data, score_threshold=0.3):
    """可视化3D检测框"""
    boxes_3d = sample_data['boxes_3d'].detach().cpu().numpy()
    scores_3d = sample_data['scores_3d'].detach().cpu().numpy()
    labels_3d = sample_data['labels_3d'].detach().cpu().numpy()
    
    # 过滤低分数检测框
    valid_mask = scores_3d >= score_threshold
    boxes_3d = boxes_3d[valid_mask]
    scores_3d = scores_3d[valid_mask]
    labels_3d = labels_3d[valid_mask]
    
    ax.set_title(f'3D目标检测 (共{len(boxes_3d)}个目标)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (前方, 米)', fontsize=12)
    ax.set_ylabel('Y (左侧, 米)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制自车
    ego_rect = Rectangle((-1, -0.9), 2, 1.8, linewidth=2, 
                         edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(ego_rect)
    ax.arrow(0, 0, 2, 0, head_width=0.5, head_length=0.5, fc='black', ec='black')
    ax.text(0, -2, 'Ego Vehicle', ha='center', fontsize=10, fontweight='bold')
    
    # 绘制检测框
    for i in range(len(boxes_3d)):
        x, y = boxes_3d[i, 0], boxes_3d[i, 1]
        w, l = boxes_3d[i, 3], boxes_3d[i, 4]  # width, length
        yaw = boxes_3d[i, 6]
        label = CLASSES[labels_3d[i]]
        score = scores_3d[i]
        color = CLASS_COLORS.get(label, 'gray')
        
        # 绘制方框（简化为矩形中心点）
        ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.7)
        
        # 绘制朝向箭头
        dx = np.cos(yaw) * l / 2
        dy = np.sin(yaw) * l / 2
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, 
                fc=color, ec=color, alpha=0.5, linewidth=1.5)
        
        # 添加标签
        if i < 10:  # 只显示前10个以免过于拥挤
            ax.text(x, y+1, f'{label}\n{score:.2f}', 
                   ha='center', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=CLASS_COLORS[cls], 
                                 markersize=8, label=cls)
                      for cls in ['car', 'truck', 'pedestrian', 'bicycle']]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)


def visualize_trajectories(ax, sample_data, score_threshold=0.3):
    """可视化目标轨迹预测"""
    boxes_3d = sample_data['boxes_3d'].detach().cpu().numpy()
    scores_3d = sample_data['scores_3d'].detach().cpu().numpy()
    labels_3d = sample_data['labels_3d'].detach().cpu().numpy()
    trajs_3d = sample_data['trajs_3d'].detach().cpu().numpy()
    trajs_score = sample_data['trajs_score'].detach().cpu().numpy()
    
    # 过滤
    valid_mask = scores_3d >= score_threshold
    boxes_3d = boxes_3d[valid_mask]
    labels_3d = labels_3d[valid_mask]
    trajs_3d = trajs_3d[valid_mask]
    trajs_score = trajs_score[valid_mask]
    
    ax.set_title(f'多模态轨迹预测 (前{min(5, len(boxes_3d))}个目标)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (前方, 米)', fontsize=12)
    ax.set_ylabel('Y (左侧, 米)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制自车
    ax.plot(0, 0, 'k*', markersize=15, label='Ego Vehicle')
    
    # 只显示前5个目标的轨迹
    for obj_idx in range(min(5, len(boxes_3d))):
        x0, y0 = boxes_3d[obj_idx, 0], boxes_3d[obj_idx, 1]
        label = CLASSES[labels_3d[obj_idx]]
        color = CLASS_COLORS.get(label, 'gray')
        
        # 绘制当前位置
        ax.plot(x0, y0, 'o', color=color, markersize=10, alpha=0.8)
        
        # 绘制多个模态轨迹（6个模态）
        obj_trajs = trajs_3d[obj_idx]  # shape: (6, 12, 2)
        obj_scores = trajs_score[obj_idx]  # shape: (6,)
        
        # 按分数排序，显示得分最高的3个模态
        top_indices = np.argsort(obj_scores)[-3:]
        
        for mode_idx in top_indices:
            traj = obj_trajs[mode_idx]  # shape: (12, 2)
            score = obj_scores[mode_idx]
            
            # 绘制轨迹
            ax.plot(traj[:, 0], traj[:, 1], '-', 
                   color=color, alpha=0.3 + 0.3 * score, linewidth=2)
            
            # 标记终点
            ax.plot(traj[-1, 0], traj[-1, 1], 's', 
                   color=color, markersize=5, alpha=0.5)
        
        # 添加标签
        ax.text(x0, y0+2, f'{label}\n{obj_idx}', 
               ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlim(-20, 80)
    ax.set_ylim(-40, 40)
    ax.legend(loc='upper left', fontsize=9)


def visualize_map(ax, sample_data, score_threshold=0.3):
    """可视化地图元素预测"""
    vectors = sample_data['vectors']
    scores = sample_data['scores']
    labels = sample_data['labels']
    
    # 过滤
    valid_mask = scores >= score_threshold
    valid_indices = np.where(valid_mask)[0]
    
    ax.set_title(f'地图元素预测 (共{len(valid_indices)}个元素)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (前方, 米)', fontsize=12)
    ax.set_ylabel('Y (左侧, 米)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制自车
    ax.plot(0, 0, 'k*', markersize=15, label='Ego Vehicle')
    
    # 绘制地图元素
    for idx in valid_indices:
        vector = vectors[idx]
        if isinstance(vector, np.ndarray):
            vector_np = vector
        else:
            vector_np = np.array(vector)
        
        label = MAP_CLASSES[labels[idx]]
        score = scores[idx]
        color = MAP_COLORS.get(label, 'gray')
        
        # 绘制折线
        ax.plot(vector_np[:, 0], vector_np[:, 1], '-', 
               color=color, linewidth=2, alpha=min(1.0, score), label=label)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    
    # 去重图例
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)


def visualize_planning(ax, sample_data):
    """可视化规划轨迹"""
    planning = sample_data['planning'].detach().cpu().numpy()  # shape: (3, 6, T, 2)
    planning_score = sample_data['planning_score'].detach().cpu().numpy()  # shape: (3, 6)
    final_planning = sample_data['final_planning'].detach().cpu().numpy()  # shape: (T, 2)
    
    gt_cmd = sample_data.get('gt_ego_fut_cmd')
    if gt_cmd is not None:
        if isinstance(gt_cmd, np.ndarray):
            cmd_idx = np.argmax(gt_cmd)
        else:
            cmd_idx = 1  # 默认直行
    else:
        cmd_idx = 1
    
    cmd_names = ['左转', '直行', '右转']
    
    ax.set_title(f'运动规划 - 当前指令: {cmd_names[cmd_idx]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (前方, 米)', fontsize=12)
    ax.set_ylabel('Y (左侧, 米)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制自车
    ego_rect = Rectangle((-1, -0.9), 2, 1.8, linewidth=2, 
                         edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(ego_rect)
    ax.arrow(0, 0, 2, 0, head_width=0.5, head_length=0.5, fc='black', ec='black')
    
    # 绘制所有指令下的多模态轨迹（淡化显示）
    colors = ['blue', 'green', 'red']
    for cmd in range(3):
        cmd_planning = planning[cmd]  # shape: (6, T, 2)
        cmd_scores = planning_score[cmd]  # shape: (6,)
        
        for mode_idx in range(len(cmd_planning)):
            traj = cmd_planning[mode_idx]
            score = cmd_scores[mode_idx]
            alpha = 0.1 if cmd != cmd_idx else 0.3 + 0.3 * score
            
            ax.plot(traj[:, 0], traj[:, 1], '-', 
                   color=colors[cmd], alpha=alpha, linewidth=1.5,
                   label=cmd_names[cmd] if mode_idx == 0 else "")
    
    # 绘制最终选定的规划轨迹
    ax.plot(final_planning[:, 0], final_planning[:, 1], 
           'o-', color='yellow', linewidth=4, markersize=6,
           alpha=0.8, label='最终规划轨迹')
    
    # 标注时间步
    for t in range(0, len(final_planning), 2):  # 每隔2个时间步标注一次
        ax.text(final_planning[t, 0], final_planning[t, 1] + 0.5, 
               f't={t}', fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(-20, 20)
    
    # 去重图例
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)


def main():
    """主函数"""
    pkl_file = "/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl"
    
    print("正在加载pkl文件...")
    predictions = load_prediction_data(pkl_file)
    print(f"✓ 成功加载，共 {len(predictions)} 个样本\n")
    
    # 选择一个样本进行可视化
    sample_idx = 100  # 可以修改这个索引来查看不同的样本
    
    if sample_idx >= len(predictions):
        print(f"警告: 样本索引 {sample_idx} 超出范围 [0, {len(predictions)-1}]")
        sample_idx = 0
    
    sample = predictions[sample_idx]
    sample_data = sample['img_bbox']
    
    print(f"正在可视化样本 #{sample_idx}...")
    print(f"Sample Token: {sample_data['token']}")
    print(f"检测目标数: {len(sample_data['boxes_3d'])}")
    print(f"地图元素数: {len(sample_data['vectors'])}")
    
    # 生成可视化
    fig = visualize_sample(sample_data, sample_idx)
    
    # 保存图片
    output_path = f"/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/scripts/sample_{sample_idx}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {output_path}")
    
    # 显示图片
    plt.show()  # 取消注释以显示图片


if __name__ == "__main__":
    main()



