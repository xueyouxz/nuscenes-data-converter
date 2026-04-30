"""
nuScenes 地图瓦片生成工具 (Final版本)

核心特点：
1. 自底向上生成瓦片（从原始分辨率到概览图）
2. 批量处理多个地图
3. 使用 WebP 格式提高性能
4. 边缘瓦片自动填充透明背景
"""

import math
import json
import time
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

# 允许加载超大图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


@dataclass
class TileConfig:
    """瓦片配置"""
    maps_path: str
    output_path: str
    tile_size: int = 256
    webp_quality: int = 100
    webp_method: int = 4


def generate_tiles(image_path: Path, map_id: str, config: TileConfig) -> None:
    """
    将大图切片为 XYZ 结构的瓦片 (非地理坐标系/笛卡尔坐标系)

    Args:
        image_path: 源图片路径
        map_id: 地图ID
        config: 瓦片配置
    """
    start_time = time.time()
    
    # 1. 加载原始图片
    print(f"Loading image: {image_path}")
    img = Image.open(image_path)

    # 确保是 RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    original_width, original_height = img.size
    print(f"Original Size: {original_width}x{original_height}")

    # 2. 计算层级 (Zoom Levels)
    # Max Zoom = 原始分辨率
    # Min Zoom (0) = 整张图缩放到能塞进一个 tile_size
    max_dim = max(original_width, original_height)
    max_zoom = math.ceil(math.log2(max_dim / config.tile_size))

    print(f"Generating Zoom Levels: 0 to {max_zoom}")

    # 创建输出目录 {output_path}/{map_id}
    output_dir = Path(config.output_path) / map_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 自底向上生成（从原始分辨率开始，逐渐缩小）
    current_img = img

    # z = max_zoom (原始清晰度) -> ... -> z = 0 (概览图)
    for z in range(max_zoom, -1, -1):
        z_dir = output_dir / str(z)
        z_dir.mkdir(parents=True, exist_ok=True)

        cw, ch = current_img.size
        cols = math.ceil(cw / config.tile_size)
        rows = math.ceil(ch / config.tile_size)

        print(f"Processing Zoom {z}: {cw}x{ch} -> {cols}x{rows} tiles")

        for x in tqdm(range(cols), desc=f"Zoom {z}", leave=False):
            # 创建 x 文件夹 (z/x/y.webp 结构)
            x_dir = z_dir / str(x)
            x_dir.mkdir(parents=True, exist_ok=True)

            for y in range(rows):
                # 计算切片区域 (Left, Top, Right, Bottom)
                left = x * config.tile_size
                top = y * config.tile_size
                right = min(left + config.tile_size, cw)
                bottom = min(top + config.tile_size, ch)

                box = (left, top, right, bottom)
                tile = current_img.crop(box)

                # 如果切片不足 tile_size (边缘)，补全透明背景
                if tile.size != (config.tile_size, config.tile_size):
                    new_tile = Image.new("RGBA", (config.tile_size, config.tile_size), (0, 0, 0, 0))
                    new_tile.paste(tile, (0, 0))
                    tile = new_tile

                # 保存为 WebP 格式
                tile_path = x_dir / f"{y}.webp"
                tile.save(
                    tile_path,
                    'WEBP',
                    quality=config.webp_quality,
                    method=config.webp_method,
                    lossless=False
                )

        # 准备下一层级：缩小 50%
        if z > 0:
            new_width = max(1, cw // 2)
            new_height = max(1, ch // 2)
            current_img = current_img.resize(
                (new_width, new_height),
                resample=Image.Resampling.LANCZOS
            )

    # 4. 生成元数据 (Metadata)
    metadata = {
        "id": map_id,
        "width": original_width,
        "height": original_height,
        "tileSize": config.tile_size,
        "minZoom": 0,
        "maxZoom": max_zoom,
        "format": "webp",
        "bounds": [0, 0, original_width, original_height],
        "createdAt": datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # 统计结果
    end_time = time.time()
    duration = end_time - start_time
    
    total_tiles = sum(1 for _ in output_dir.rglob("*.webp"))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*.webp"))
    
    print(f"✓ {map_id} 完成: {total_tiles} 个瓦片, "
          f"{total_size / 1024 / 1024:.1f} MB, 耗时 {duration:.1f}s")


def process_all_maps(config: TileConfig) -> None:
    """处理所有地图"""
    print("=" * 60)
    print("nuScenes 地图瓦片生成工具 (Final版)")
    print("=" * 60)
    print(f"输入: {config.maps_path}")
    print(f"输出: {config.output_path}")
    print(f"配置: {config.tile_size}×{config.tile_size} WebP (质量:{config.webp_quality})")
    print("=" * 60)

    maps_dir = Path(config.maps_path)
    png_files = [
        f for f in maps_dir.glob("*.png")
        if not f.name.endswith('.aux.xml')
    ]

    if not png_files:
        print("未找到地图文件！")
        return

    print(f"\n开始处理 {len(png_files)} 个地图...\n")

    for png_file in png_files:
        map_id = png_file.stem
        generate_tiles(png_file, map_id, config)

    print("\n" + "=" * 60)
    print("✓ 所有地图处理完成！")
    print("=" * 60)


def main():
    """主函数"""
    config = TileConfig(
        maps_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/basemap",
        output_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/tiles_final",
        tile_size=256,
        webp_quality=100,
        webp_method=4
    )

    try:
        process_all_maps(config)
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n致命错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()