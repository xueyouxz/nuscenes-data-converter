"""
NuScenes数据转换器（优化版本）

版本 2.0.0 - 性能优化版本

主要优化：
1. 引入计算流水线（ScenePipeline）消除重复计算
2. 提取通用匹配和距离计算模块
3. 优化Chamfer距离批量计算（O(n²) → O(n)）
4. 支持多线程批量转换
"""

from .converter import (
    DataConverter,
)
from .config import config, Config
from .core.pipeline import ScenePipeline

__version__ = "2.0.0"
__all__ = [
    'DataConverter',
    'ScenePipeline',
    'config',
    'Config'
]

