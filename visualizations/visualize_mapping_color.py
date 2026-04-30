import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_mapping_color(file_path):
    """加载地图着色数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def visualize_mapping_color(mapping_color_path, output_path=None, colormap='RdYlGn_r'):
    """
    可视化地图着色数据，根据预测误差映射颜色
    
    Args:
        mapping_color_path: 地图着色文件路径
        output_path: 可选，保存图片的路径
        colormap: 颜色映射方案（默认：RdYlGn_r，红色表示高误差，绿色表示低误差）
    """
    data = load_mapping_color(mapping_color_path)
    
    colored_elements = data.get('colored_elements', {})
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    all_errors = []
    for category in ['divider', 'boundary', 'ped_crossing']:
        elements = colored_elements.get(category, [])
        for element in elements:
            point_errors = element.get('point_errors', [])
            all_errors.extend(point_errors)
    
    if not all_errors:
        print("警告: 没有找到误差数据")
        return
    
    min_error = min(all_errors)
    max_error = max(all_errors)
    
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=min_error, vmax=max_error)
    
    category_styles = {
        'divider': {'linewidth': 2.0, 'alpha': 0.8},
        'boundary': {'linewidth': 2.5, 'alpha': 0.8},
        'ped_crossing': {'linewidth': 2.0, 'alpha': 0.7}
    }
    
    category_labeled = {'divider': False, 'boundary': False, 'ped_crossing': False}
    
    for category in ['divider', 'boundary', 'ped_crossing']:
        elements = colored_elements.get(category, [])
        style = category_styles[category]
        
        for element in elements:
            coordinates = element.get('coordinates', [])
            point_errors = element.get('point_errors', [])
            
            if not coordinates:
                continue
            
            coords = np.array(coordinates)
            
            if len(point_errors) == len(coords):
                for i in range(len(coords) - 1):
                    error = (point_errors[i] + point_errors[i + 1]) / 2
                    color = cmap(norm(error))
                    ax.plot(coords[i:i+2, 0], coords[i:i+2, 1], 
                           color=color, linewidth=style['linewidth'], 
                           alpha=style['alpha'],
                           label=category.capitalize() if not category_labeled[category] else '')
                    category_labeled[category] = True
            else:
                avg_error = element.get('avg_error', 0)
                color = cmap(norm(avg_error))
                ax.plot(coords[:, 0], coords[:, 1], 
                       color=color, linewidth=style['linewidth'], 
                       alpha=style['alpha'],
                       label=category.capitalize() if not category_labeled[category] else '')
                category_labeled[category] = True
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Prediction Error (m)', fontsize=12)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Mapping Color Visualization\n(Color indicates prediction error)', 
                 fontsize=14, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='可视化地图着色数据')
    parser.add_argument('--mapping_color', type=str,
                       default='../output/scene-0103/mapping_color.json',
                       help='地图着色文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（可选，如果不指定则显示）')
    parser.add_argument('--colormap', type=str, default='RdYlGn_r',
                       help='颜色映射方案（默认: RdYlGn_r）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mapping_color):
        print(f"错误: 地图着色文件不存在: {args.mapping_color}")
        return
    
    visualize_mapping_color(args.mapping_color, args.output, args.colormap)

if __name__ == '__main__':
    main()

