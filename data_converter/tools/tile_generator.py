"""
nuScenes 地图瓦片生成工具 (多线程版本)

将 nuScenes 大地图切片为瓦片金字塔，生成标准的 XYZ 瓦片结构，使用 WebP 格式
瓦片结构: {output_path}/{map_id}/{z}/{x}/{y}.webp
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from PIL import Image, ImageFile
from datetime import datetime
import threading

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
    webp_method: int = 4  # 0-6，值越高压缩率越高但速度越慢
    max_workers: int = 8  # 最大线程数


class TileGenerator:
    """地图瓦片生成器"""
    
    def __init__(self, config: TileConfig):
        self.config = config
        self.lock = threading.Lock()  # 用于线程安全的进度输出
        
    def calculate_max_zoom(self, width: int, height: int) -> int:
        """
        计算图片金字塔的最大层级
        
        Args:
            width: 图片宽度
            height: 图片高度
            
        Returns:
            最大缩放层级
        """
        max_dimension = max(width, height)
        return math.ceil(math.log2(max_dimension / self.config.tile_size))
    
    def generate_tiles(self, input_file: Path, map_id: str) -> None:
        """
        为单个地图生成瓦片金字塔
        
        Args:
            input_file: 输入文件路径
            map_id: 地图ID
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"处理地图: {map_id}")
        print(f"{'='*60}")
        
        try:
            # 读取图片
            with Image.open(input_file) as img:
                width, height = img.size
                file_size = input_file.stat().st_size / 1024 / 1024  # MB
                
                print(f"原始尺寸: {width} x {height}")
                print(f"文件大小: {file_size:.2f} MB")
                
                # 计算层级
                max_zoom = self.calculate_max_zoom(width, height)
                print(f"瓦片层级: 0 - {max_zoom}")
                print(f"瓦片尺寸: {self.config.tile_size}x{self.config.tile_size}")
                
                # 创建输出目录
                map_output_path = Path(self.config.output_path) / map_id
                map_output_path.mkdir(parents=True, exist_ok=True)
                
                # 保存元数据
                tile_metadata = {
                    "id": map_id,
                    "originalWidth": width,
                    "originalHeight": height,
                    "tileSize": self.config.tile_size,
                    "minZoom": 0,
                    "maxZoom": max_zoom,
                    "format": "webp",
                    "bounds": [0, 0, width, height],
                    "createdAt": datetime.now().isoformat(),
                }
                
                metadata_file = map_output_path / "metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(tile_metadata, f, indent=2)
                print("✓ 已保存元数据")
                
                # 生成每个层级的瓦片
                for z in range(max_zoom + 1):
                    self.generate_zoom_level(input_file, map_output_path, z, max_zoom, width, height)
                
            # 统计结果
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n✓ 完成! 总耗时: {duration:.2f}s")
            
            stats = self.get_tile_stats(map_output_path)
            print(f"\n瓦片统计:")
            print(f"  - 总瓦片数: {stats['total_tiles']}")
            print(f"  - 总大小: {stats['total_size'] / 1024 / 1024:.2f} MB")
            if stats['total_tiles'] > 0:
                print(f"  - 平均瓦片大小: {stats['total_size'] / stats['total_tiles'] / 1024:.2f} KB")
                
        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            raise
    
    def generate_zoom_level(
        self, 
        input_file: Path, 
        output_path: Path, 
        z: int, 
        max_zoom: int,
        original_width: int,
        original_height: int
    ) -> None:
        """
        生成指定缩放层级的所有瓦片（多线程版本）
        
        Args:
            input_file: 输入文件路径
            output_path: 输出目录路径
            z: 当前层级
            max_zoom: 最大层级
            original_width: 原始图片宽度
            original_height: 原始图片高度
        """
        print(f"\n生成层级 z={z}...")
        
        # 计算此层级的图片尺寸
        scale = math.pow(2, z - max_zoom)
        level_width = math.ceil(original_width * scale)
        level_height = math.ceil(original_height * scale)
        
        print(f"  层级尺寸: {level_width} x {level_height}")
        
        # 计算此层级的瓦片数量
        tiles_x = math.ceil(level_width / self.config.tile_size)
        tiles_y = math.ceil(level_height / self.config.tile_size)
        total_tiles = tiles_x * tiles_y
        
        print(f"  瓦片数量: {tiles_x} x {tiles_y} = {total_tiles} 个")
        
        # 先缩放整个图片到此层级
        with Image.open(input_file) as img:
            # 使用高质量缩放算法
            resized_img = img.resize(
                (level_width, level_height),
                Image.Resampling.LANCZOS
            )
            
            # 生成所有瓦片任务
            tasks = []
            for y in range(tiles_y):
                for x in range(tiles_x):
                    tasks.append((x, y, level_width, level_height))
            
            # 使用多线程处理瓦片
            processed_tiles = 0
            progress_step = max(1, total_tiles // 10)  # 每10%输出一次
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(
                        self.generate_tile,
                        resized_img.copy(),
                        output_path,
                        z,
                        x,
                        y,
                        level_width,
                        level_height
                    ): (x, y)
                    for x, y, level_width, level_height in tasks
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_task):
                    try:
                        future.result()
                        processed_tiles += 1
                        
                        # 输出进度
                        if processed_tiles % progress_step == 0 or processed_tiles == total_tiles:
                            progress = (processed_tiles / total_tiles) * 100
                            with self.lock:
                                print(f"  进度: {progress:.0f}% ({processed_tiles}/{total_tiles})")
                    except Exception as e:
                        x, y = future_to_task[future]
                        print(f"  ✗ 瓦片 ({z}/{x}/{y}) 生成失败: {str(e)}")
        
        print(f"  ✓ 层级 z={z} 完成")
    
    def generate_tile(
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
        生成单个瓦片
        
        Args:
            level_img: 当前层级的完整图片
            output_path: 输出目录路径
            z: 缩放层级
            x: X坐标
            y: Y坐标
            level_width: 层级图片宽度
            level_height: 层级图片高度
        """
        try:
            # 计算裁剪区域
            left = x * self.config.tile_size
            top = y * self.config.tile_size
            width = min(self.config.tile_size, level_width - left)
            height = min(self.config.tile_size, level_height - top)
            
            # 裁剪瓦片
            tile = level_img.crop((left, top, left + width, top + height))
            
            # 如果瓦片不足标准尺寸，扩展为标准尺寸（填充透明背景）
            if width < self.config.tile_size or height < self.config.tile_size:
                new_tile = Image.new(
                    'RGBA',
                    (self.config.tile_size, self.config.tile_size),
                    (0, 0, 0, 0)
                )
                new_tile.paste(tile, (0, 0))
                tile = new_tile
            
            # 确保是RGBA模式
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
            
        finally:
            # 关闭图片以释放内存
            if level_img:
                level_img.close()
    
    def get_tile_stats(self, output_path: Path) -> Dict[str, Any]:
        """
        统计瓦片信息
        
        Args:
            output_path: 输出目录路径
            
        Returns:
            统计信息字典
        """
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
        print("nuScenes 地图瓦片生成工具 (多线程版本)")
        print("="*60)
        print(f"输入目录: {self.config.maps_path}")
        print(f"输出目录: {self.config.output_path}")
        print(f"瓦片尺寸: {self.config.tile_size}x{self.config.tile_size}")
        print(f"图片格式: WebP (质量: {self.config.webp_quality})")
        print(f"线程数: {self.config.max_workers}")
        
        # 创建输出目录
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取所有 PNG 文件
        maps_dir = Path(self.config.maps_path)
        png_files = [
            f for f in maps_dir.glob("*.png")
            if not f.name.endswith('.aux.xml')
        ]
        
        print(f"\n找到 {len(png_files)} 个地图文件\n")
        
        # 处理每个地图
        for png_file in png_files:
            map_id = png_file.stem
            self.generate_tiles(png_file, map_id)
        
        print(f"\n{'='*60}")
        print("所有地图处理完成！")
        print(f"{'='*60}")


def main():
    """主函数"""
    # 配置（根据您的实际路径修改）
    config = TileConfig(
        maps_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/basemap",  # 修改为实际路径
        output_path="/home/public/nuscenes_datasets/nuscenes-mini/maps/tiles",  # 修改为实际路径
        tile_size=256,
        webp_quality=100,
        webp_method=4,
        max_workers=8  # 根据CPU核心数调整
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


