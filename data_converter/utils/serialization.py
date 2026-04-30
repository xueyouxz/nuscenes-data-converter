"""
MessagePack序列化工具模块

提供数据序列化和反序列化功能，使用MessagePack格式进行二进制序列化
相比JSON更高效，体积更小，速度更快
"""

import msgpack
import numpy as np
from typing import Any, Dict, List, Union
from pathlib import Path


def serialize_to_msgpack(data: Any) -> bytes:
    """
    将数据序列化为MessagePack格式的字节流
    
    Args:
        data: 需要序列化的数据，支持dict、list、numpy数组等
        
    Returns:
        序列化后的字节流
        
    原理：
        MessagePack是一种高效的二进制序列化格式
        1. 自动处理numpy数组，转换为列表
        2. 支持嵌套的字典和列表结构
        3. 使用use_bin_type=True确保二进制数据正确处理
    """
    # 预处理数据，将numpy数组转换为列表
    processed_data = _preprocess_for_msgpack(data)
    
    # 序列化为MessagePack格式
    packed_bytes = msgpack.packb(processed_data, use_bin_type=True)
    
    return packed_bytes


def deserialize_from_msgpack(packed_bytes: bytes) -> Any:
    """
    从MessagePack字节流反序列化数据
    
    Args:
        packed_bytes: MessagePack格式的字节流
        
    Returns:
        反序列化后的数据
        
    原理：
        使用raw=False确保字符串正确解码为str类型而不是bytes
    """
    data = msgpack.unpackb(packed_bytes, raw=False)
    return data


def save_msgpack_file(data: Any, file_path: Union[str, Path]) -> None:
    """
    将数据序列化并保存到.bin文件
    
    Args:
        data: 需要保存的数据
        file_path: 保存路径，建议使用.bin扩展名
        
    原理：
        1. 先序列化为MessagePack字节流
        2. 以二进制模式写入文件
        3. 自动创建父目录（如果不存在）
    """
    file_path = Path(file_path)
    
    # 确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 序列化数据
    packed_bytes = serialize_to_msgpack(data)
    
    # 写入文件
    with open(file_path, 'wb') as f:
        f.write(packed_bytes)


def load_msgpack_file(file_path: Union[str, Path]) -> Any:
    """
    从.bin文件加载并反序列化数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        反序列化后的数据
        
    原理：
        1. 以二进制模式读取文件
        2. 反序列化MessagePack字节流
    """
    file_path = Path(file_path)
    
    # 读取文件
    with open(file_path, 'rb') as f:
        packed_bytes = f.read()
    
    # 反序列化
    data = deserialize_from_msgpack(packed_bytes)
    
    return data


def _preprocess_for_msgpack(obj: Any) -> Any:
    """
    预处理数据以适配MessagePack序列化
    
    Args:
        obj: 需要处理的对象
        
    Returns:
        处理后的对象
        
    原理：
        递归处理数据结构，将不支持的类型转换为MessagePack支持的类型
        - numpy数组 -> 列表
        - numpy标量 -> Python标量
        - 字典和列表递归处理
    """
    # None、布尔值、字符串、数字直接返回
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    
    # numpy标量类型转换
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # numpy数组转换为列表
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # 字典递归处理
    if isinstance(obj, dict):
        return {str(k): _preprocess_for_msgpack(v) for k, v in obj.items()}
    
    # 列表和元组递归处理
    if isinstance(obj, (list, tuple)):
        return [_preprocess_for_msgpack(item) for item in obj]
    
    # 其他类型尝试转换为字符串
    return str(obj)


def convert_numpy_to_list(data: Union[Dict, List, np.ndarray, Any]) -> Any:
    """
    将数据中的所有numpy数组转换为列表
    这是一个更通用的辅助函数，可以在序列化前使用
    
    Args:
        data: 包含numpy数组的数据结构
        
    Returns:
        所有numpy数组都转换为列表的数据结构
    """
    return _preprocess_for_msgpack(data)

