"""
nuScenes 地图瓦片生成工具 (无填充版本)

核心特点：
1. 直接使用原始图片尺寸，不进行填充
2. 各层级之间是严格的 2 倍缩放关系
3. 边缘瓦片可能不完整（尺寸 < tileSize）
4. 保留地理参考信息（坐标系统、bounds、分辨率）以支持前端对齐

瓦片结构: {output_path}/{map_id}/{z}/{x}/{y}.webp
坐标系统: nuScenes全局坐标系（米）
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from PIL import Image, ImageFile
from datetime import datetime
from tqdm import tqdm

# 允许加载超大图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


@dataclass
class TileConfig:
    """瓦片配置"""
    maps_path: str
    output_path: str
    tile_size: int = 128
    webp_quality: int = 100
    webp_method: int = 4
    max_workers: int = 16
    # 可选：basemap元数据路径（用于获取地理参考信息）
    basemap_metadata_path: Optional[str] = None


class TileGenerator:
    """地图瓦片生成器（优化版）"""

    def __init__(self, config: TileConfig):
        self.config = config

    def calculate_max_zoom(self, width: int, height: int) -> int:
        """
        计算最大缩放层级（基于原始尺寸）

        策略：找到最小的层级，使得该层级的图像尺寸不小于一个瓦片

        Args:
            width: 原始宽度
            height: 原始高度

        Returns:
            max_zoom: 最大缩放层级
        """
        max_dimension = max(width, height)

        # 计算需要多少层
        # 最小层级（z=0）时，图像至少要有一个瓦片的大小
        # max_dimension / 2^max_zoom >= tile_size
        # 2^max_zoom <= max_dimension / tile_size
        # max_zoom <= log2(max_dimension / tile_size)
        max_zoom = math.floor(math.log2(max_dimension / self.config.tile_size))
        max_zoom = max(0, max_zoom)  # 确保至少为 0

        return max_zoom


    def generate_tiles(self, input_file: Path, map_id: str) -> None:
        """
        为单个地图生成瓦片金字塔

        Args:
            input_file: 输入文件路径
            map_id: 地图ID
        """
        start_time = time.time()

        try:
            # 尝试加载basemap元数据（如果存在）
            basemap_metadata = self._load_basemap_metadata(input_file, map_id)
            
            with Image.open(input_file) as img:
                # 转换为 RGBA（如果需要）
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                original_width, original_height = img.size

                # 计算最大缩放层级
                max_zoom = self.calculate_max_zoom(original_width, original_height)

                # 创建输出目录
                map_output_path = Path(self.config.output_path) / map_id
                map_output_path.mkdir(parents=True, exist_ok=True)

                # 构建完整的元数据（包含地理参考信息）
                tile_metadata = self._build_tile_metadata(
                    map_id,
                    original_width,
                    original_height,
                    max_zoom,
                    basemap_metadata
                )

                metadata_file = map_output_path / "metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(tile_metadata, f, indent=2)

                # 生成每个层级的瓦片（使用进度条）
                for z in tqdm(range(max_zoom + 1), desc=f"处理 {map_id}", unit="层级"):
                    self.generate_zoom_level(
                        img,
                        map_output_path,
                        z,
                        max_zoom,
                        original_width,
                        original_height
                    )

            # 统计结果
            end_time = time.time()
            duration = end_time - start_time

            stats = self.get_tile_stats(map_output_path)
            print(f"✓ {map_id} 完成: {stats['total_tiles']} 个瓦片, "
                  f"{stats['total_size'] / 1024 / 1024:.1f} MB, 耗时 {duration:.1f}s")

        except Exception as e:
            print(f"✗ {map_id} 错误: {str(e)}")
            raise

    def _load_basemap_metadata(self, input_file: Path, map_id: str) -> Optional[Dict[str, Any]]:
        """
        加载basemap元数据（如果存在）
        
        优先级：
        1. 配置中指定的basemap_metadata_path
        2. 输入文件同目录下的basemap_metadata.json
        3. 返回None（无地理参考信息）
        
        Args:
            input_file: 输入文件路径
            map_id: 地图ID
            
        Returns:
            basemap元数据字典或None
        """
        metadata_paths = []
        
        # 优先级1：配置中指定的路径
        if self.config.basemap_metadata_path:
            metadata_paths.append(Path(self.config.basemap_metadata_path))
        
        # 优先级2：输入文件同目录下
        metadata_paths.append(input_file.parent / "basemap_metadata.json")
        
        # 优先级3：以map_id命名的元数据文件
        metadata_paths.append(input_file.parent / f"{map_id}_metadata.json")
        
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        print(f"  ✓ 加载地理参考信息: {metadata_path.name}")
                        return metadata
                except Exception as e:
                    print(f"  ⚠ 无法加载 {metadata_path.name}: {str(e)}")
        
        print("  ℹ 未找到地理参考信息，使用像素坐标系统")
        return None

    def _build_tile_metadata(
        self,
        map_id: str,
        width: int,
        height: int,
        max_zoom: int,
        basemap_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建瓦片元数据（包含地理参考信息）
        
        Args:
            map_id: 地图ID
            width: 图片宽度（像素）
            height: 图片高度（像素）
            max_zoom: 最大缩放层级
            basemap_metadata: basemap元数据（可选）
            
        Returns:
            完整的瓦片元数据
        """
        # 基础元数据
        metadata = {
            "id": map_id,
            "version": "2.0",
            "tileSize": self.config.tile_size,
            "minZoom": 0,
            "maxZoom": max_zoom,
            "format": "webp",
            "createdAt": datetime.now().isoformat(),
            
            # 图像信息
            "image": {
                "width": width,
                "height": height,
                "format": "PNG"
            },
            
            # 像素坐标系边界（始终提供）
            "pixelBounds": {
                "minX": 0,
                "maxX": width,
                "minY": 0,
                "maxY": height
            }
        }
        
        # 如果有basemap元数据，添加地理参考信息
        if basemap_metadata:
            # 提取关键信息
            bounds = basemap_metadata.get('bounds', {})
            resolution = basemap_metadata.get('resolution', {})
            coord_system = basemap_metadata.get('coordinate_system', {})
            transform = basemap_metadata.get('transform', {})
            
            # 添加地理参考信息
            metadata["georeference"] = {
                # 全局坐标系边界（米）
                "bounds": {
                    "minX": bounds.get('min_x'),
                    "maxX": bounds.get('max_x'),
                    "minY": bounds.get('min_y'),
                    "maxY": bounds.get('max_y'),
                    "widthMeters": bounds.get('width_meters'),
                    "heightMeters": bounds.get('height_meters')
                },
                
                # 分辨率信息
                "resolution": {
                    "metersPerPixelX": resolution.get('meters_per_pixel_x'),
                    "metersPerPixelY": resolution.get('meters_per_pixel_y'),
                    "pixelsPerMeter": resolution.get('pixels_per_meter')
                },
                
                # 坐标系统信息
                "coordinateSystem": {
                    "type": coord_system.get('type', 'global'),
                    "unit": coord_system.get('unit', 'meters'),
                    "description": coord_system.get('description', 'nuScenes全局坐标系')
                },
                
                # 坐标变换信息
                "transform": {
                    "originX": transform.get('origin_x', bounds.get('min_x')),
                    "originY": transform.get('origin_y', bounds.get('max_y')),
                    "yAxisDirection": transform.get('y_axis_direction', 'down'),
                    "notes": transform.get('notes', '图像原点在左上角，Y轴向下；全局坐标Y轴向上')
                }
            }
            
            # 添加坐标转换公式说明（供前端参考）
            metadata["coordinateConversion"] = {
                "description": "坐标转换公式",
                "pixelToGlobal": {
                    "formula": "globalX = minX + pixelX * metersPerPixelX; globalY = maxY - pixelY * metersPerPixelY",
                    "example": {
                        "input": {"pixelX": 0, "pixelY": 0},
                        "output": {
                            "globalX": bounds.get('min_x'),
                            "globalY": bounds.get('max_y')
                        }
                    }
                },
                "globalToPixel": {
                    "formula": "pixelX = (globalX - minX) / metersPerPixelX; pixelY = (maxY - globalY) / metersPerPixelY",
                    "example": {
                        "input": {
                            "globalX": bounds.get('min_x'),
                            "globalY": bounds.get('max_y')
                        },
                        "output": {"pixelX": 0, "pixelY": 0}
                    }
                }
            }
        else:
            # 没有地理参考信息时的说明
            metadata["georeference"] = None
            metadata["note"] = "此瓦片集未包含地理参考信息，仅支持像素坐标系统"
        
        return metadata

    def generate_zoom_level(
            self,
            original_img: Image.Image,
            output_path: Path,
            z: int,
            max_zoom: int,
            original_width: int,
            original_height: int
    ) -> None:
        """
        生成指定缩放层级的所有瓦片

        策略：
        1. 计算当前层级的缩放比例（相对于最大层级）
        2. 缩放原图到当前层级尺寸
        3. 切割瓦片，边缘瓦片可能不完整

        Args:
            original_img: 原始图片
            output_path: 输出目录
            z: 当前层级
            max_zoom: 最大层级
            original_width: 原始宽度
            original_height: 原始高度
        """
        # 计算此层级的缩放比例
        # z=max_zoom 时，scale=1 (原始尺寸)
        # z=0 时，scale=1/2^max_zoom (最小尺寸)
        scale = 2 ** (z - max_zoom)
        
        # 计算此层级的图像尺寸
        level_width = max(1, int(original_width * scale))
        level_height = max(1, int(original_height * scale))

        # 计算瓦片数量（向上取整，包含不完整的边缘瓦片）
        tiles_x = math.ceil(level_width / self.config.tile_size)
        tiles_y = math.ceil(level_height / self.config.tile_size)
        total_tiles = tiles_x * tiles_y

        # 缩放图片到当前层级尺寸
        resized_img = original_img.resize(
            (level_width, level_height),
            Image.Resampling.LANCZOS
        )

        # 生成所有瓦片任务
        tasks = [(x, y) for y in range(tiles_y) for x in range(tiles_x)]

        # 使用 tqdm 显示瓦片生成进度
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.generate_tile_from_level,
                    resized_img,
                    output_path,
                    z, x, y,
                    level_width,
                    level_height
                ): (x, y)
                for x, y in tasks
            }

            with tqdm(total=total_tiles, desc=f"  z={z} ({tiles_x}×{tiles_y})",
                      unit="瓦片", leave=False) as pbar:
                for future in as_completed(future_to_task):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        x, y = future_to_task[future]
                        tqdm.write(f"  ✗ 瓦片 ({z}/{x}/{y}) 生成失败: {str(e)}")

        # 关闭缩放后的图片
        resized_img.close()

    def generate_tile_from_level(
            self,
            level_img: Image.Image,
            output_path: Path,
            z: int,
            x: int,
            y: int,
            level_width: int,
            level_height: int
    ) -> None:
        """
        从层级图片中裁剪单个瓦片

        处理边缘瓦片：如果瓦片区域超出图像边界，只裁剪实际存在的部分

        Args:
            level_img: 当前层级的完整图片
            output_path: 输出目录
            z: 层级
            x: X坐标
            y: Y坐标
            level_width: 层级图像宽度
            level_height: 层级图像高度
        """
        tile_size = self.config.tile_size

        # 计算裁剪区域
        left = x * tile_size
        top = y * tile_size
        right = min(left + tile_size, level_width)
        bottom = min(top + tile_size, level_height)

        # 裁剪瓦片（实际尺寸可能小于 tile_size）
        tile = level_img.crop((left, top, right, bottom))
        
        # 获取实际裁剪的尺寸
        actual_width = right - left
        actual_height = bottom - top

        # 如果瓦片小于标准尺寸，需要填充到 tile_size x tile_size
        if actual_width < tile_size or actual_height < tile_size:
            # 创建透明背景的标准尺寸瓦片
            padded_tile = Image.new('RGBA', (tile_size, tile_size), (0, 0, 0, 0))
            # 将实际内容粘贴到左上角
            padded_tile.paste(tile, (0, 0))
            tile.close()
            tile = padded_tile

        # 确保是 RGBA 模式
        if tile.mode != 'RGBA':
            tile = tile.convert('RGBA')

        # 创建目录
        tile_dir = output_path / str(z) / str(x)
        tile_dir.mkdir(parents=True, exist_ok=True)

        # 保存瓦片
        tile_path = tile_dir / f"{y}.webp"
        tile.save(
            tile_path,
            'WEBP',
            quality=self.config.webp_quality,
            method=self.config.webp_method,
            lossless=False
        )

        tile.close()

    def get_tile_stats(self, output_path: Path) -> Dict[str, Any]:
        """统计瓦片信息"""
        total_tiles = 0
        total_size = 0

        for webp_file in output_path.rglob("*.webp"):
            total_tiles += 1
            total_size += webp_file.stat().st_size

        return {
            "total_tiles": total_tiles,
            "total_size": total_size
        }

    def process_all_maps(self) -> None:
        """处理所有地图"""
        print("=" * 60)
        print("nuScenes 地图瓦片生成工具 (无填充版)")
        print("=" * 60)
        print(f"输入: {self.config.maps_path}")
        print(f"输出: {self.config.output_path}")
        print(
            f"配置: {self.config.tile_size}×{self.config.tile_size} WebP (质量:{self.config.webp_quality}, 线程:{self.config.max_workers})")
        print("=" * 60)

        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        maps_dir = Path(self.config.maps_path)
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
            self.generate_tiles(png_file, map_id)

        print("\n" + "=" * 60)
        print("✓ 所有地图处理完成！")
        print("=" * 60)


def main():
    """主函数"""
    config = TileConfig(
        maps_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/basemap",
        output_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/tiles",
        tile_size=128,
        webp_quality=100,
        webp_method=4,
        max_workers=16
    )

    try:
        generator = TileGenerator(config)
        generator.process_all_maps()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n致命错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
