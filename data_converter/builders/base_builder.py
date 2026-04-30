"""
Builder基类

提供所有builder的通用功能，包括：
- 统一的文件保存逻辑
- 场景子目录创建
- 支持pipeline和传统接口的抽象基类
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path

from ..utils.serialization import save_msgpack_file


class BaseBuilder(ABC):
    """
    Builder基类
    
    所有数据构建器的抽象基类，提供统一的保存接口
    支持两种构建模式：
    1. 传统模式：build(scene_token) - 向后兼容
    2. Pipeline模式：build_from_pipeline(pipeline) - 优化性能
    """
    
    @abstractmethod
    def build(self, scene_token_or_pipeline: Union[str, 'ScenePipeline']) -> Dict[str, Any]:
        """
        构建数据（子类必须实现）
        
        Args:
            scene_token_or_pipeline: 场景token（传统模式）或 ScenePipeline（优化模式）
            
        Returns:
            构建的数据字典
            
        说明：
            子类可以检测参数类型来选择实现方式：
            - 如果是str，使用传统方式
            - 如果是ScenePipeline，使用优化方式
        """
        pass
    
    def _get_scene_dir(
        self,
        output_dir: Path,
        scene_name: str
    ) -> Path:
        """
        获取场景子目录，如果不存在则创建
        
        Args:
            output_dir: 输出根目录
            scene_name: 场景名称
            
        Returns:
            场景子目录路径
        """
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir
    
    def save_json(
        self,
        data: Dict[str, Any],
        output_dir: Path,
        scene_name: str,
        filename: str
    ) -> Path:
        """
        保存数据为JSON文件
        
        Args:
            data: 要保存的数据
            output_dir: 输出目录
            scene_name: 场景名称
            filename: 文件名（如 'metadata.json'）
            
        Returns:
            保存的文件路径
        """
        scene_dir = self._get_scene_dir(output_dir, scene_name)
        file_path = scene_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def save_msgpack(
        self,
        data: Dict[str, Any],
        output_dir: Path,
        scene_name: str,
        filename: str
    ) -> Path:
        """
        保存数据为MessagePack文件（.bin）
        
        Args:
            data: 要保存的数据
            output_dir: 输出目录
            scene_name: 场景名称
            filename: 文件名（如 'static_map.bin'）
            
        Returns:
            保存的文件路径
        """
        scene_dir = self._get_scene_dir(output_dir, scene_name)
        file_path = scene_dir / filename
        
        save_msgpack_file(data, file_path)
        
        return file_path


