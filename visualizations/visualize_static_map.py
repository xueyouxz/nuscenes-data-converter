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

def visualize_static_map(static_map_path, output_path=None):
    """
    可视化静态地图
    
    Args:
        static_map_path: 静态地图文件路径
        output_path: 可选，保存图片的路径
    """
    data = load_static_map(static_map_path)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    dividers = data.get('divider', [])
    boundaries = data.get('boundary', [])
    ped_crossings = data.get('ped_crossing', [])
    drivable_areas = data.get('drivable_area', [])
    
    divider_labeled = False
    boundary_labeled = False
    crossing_labeled = False
    drivable_labeled = False
    
    for drivable_area in drivable_areas:
        if drivable_area:
            coords = np.array(drivable_area)
            ax.fill(coords[:, 0], coords[:, 1], 'lightgray', alpha=0.3, 
                   label='Drivable Area' if not drivable_labeled else '')
            drivable_labeled = True
    
    for divider in dividers:
        if divider:
            coords = np.array(divider)
            ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.5, alpha=0.8, 
                   label='Divider' if not divider_labeled else '')
            divider_labeled = True
    
    for boundary in boundaries:
        if boundary:
            coords = np.array(boundary)
            ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2, alpha=0.8, 
                   label='Boundary' if not boundary_labeled else '')
            boundary_labeled = True
    
    for crossing in ped_crossings:
        if crossing:
            coords = np.array(crossing)
            ax.plot(coords[:, 0], coords[:, 1], 'y-', linewidth=2, alpha=0.6, 
                   label='Ped Crossing' if not crossing_labeled else '')
            crossing_labeled = True
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Static Map Visualization', fontsize=14, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='可视化静态地图')
    parser.add_argument('--static_map', type=str, 
                       default='../output/scene-0103/static_map.bin',
                       help='静态地图文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，如果不指定则显示）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.static_map):
        print(f"错误: 静态地图文件不存在: {args.static_map}")
        return
    
    visualize_static_map(args.static_map, args.output)

if __name__ == '__main__':
    main()

