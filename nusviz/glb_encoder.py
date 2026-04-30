"""
GLB 编码工具

按照 glTF 2.0 规范将数据编码为 JSON + BIN chunk 的单文件 .glb 格式。
"""

import struct
import json
from typing import Dict, Any, List, Tuple
import numpy as np


class GLBEncoder:
    """将 nuviz 消息数据编码为 GLB 文件。"""

    # glTF 2.0 magic numbers
    GLB_MAGIC = 0x46546C67   # "glTF"
    GLB_VERSION = 2
    JSON_CHUNK_TYPE = 0x4E4F534A  # "JSON"
    BIN_CHUNK_TYPE  = 0x004E4942  # "BIN\0"

    # glTF componentType 枚举
    BYTE           = 5120
    UNSIGNED_BYTE  = 5121
    SHORT          = 5122
    UNSIGNED_SHORT = 5123
    UNSIGNED_INT   = 5125
    FLOAT          = 5126

    # dtype -> componentType 映射
    _DTYPE_TO_COMPONENT_TYPE = {
        np.dtype('float32'): FLOAT,
        np.dtype('uint32'):  UNSIGNED_INT,
        np.dtype('uint16'):  UNSIGNED_SHORT,
        np.dtype('uint8'):   UNSIGNED_BYTE,
        np.dtype('int16'):   SHORT,
    }

    # 列数 -> glTF accessor type 映射
    _COLS_TO_TYPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}

    def __init__(self):
        self.buffer_views: List[Dict[str, Any]] = []
        self.accessors:    List[Dict[str, Any]] = []
        self.images:       List[Dict[str, Any]] = []
        self.bin_data:     bytearray = bytearray()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_bin(self, data: bytes, alignment: int = 4) -> Tuple[int, int]:
        """将 data 追加到 BIN chunk（含对齐填充），返回 (byteOffset, byteLength)。"""
        padding = (alignment - (len(self.bin_data) % alignment)) % alignment
        self.bin_data.extend(b'\x00' * padding)

        byte_offset = len(self.bin_data)
        self.bin_data.extend(data)
        return byte_offset, len(data)

    def _add_buffer_view(self, data: bytes) -> int:
        """将 data 写入 BIN chunk 并记录 bufferView，返回其索引。"""
        byte_offset, byte_length = self._append_bin(data)
        idx = len(self.buffer_views)
        self.buffer_views.append({"byteOffset": byte_offset, "byteLength": byte_length})
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_binary_data(self, data: bytes, alignment: int = 4) -> Tuple[int, int]:
        """
        添加原始二进制数据到 BIN chunk。

        Returns:
            (byteOffset, byteLength)
        """
        return self._append_bin(data, alignment)

    def add_accessor(
        self,
        data: np.ndarray,
        component_type: int = None,
        type_str: str = None,
        normalized: bool = False,
    ) -> int:
        """
        将 numpy 数组写入 GLB 并创建对应的 accessor，返回其索引。

        component_type 和 type_str 均可自动从 data.dtype / data.shape 推断。
        """
        # 推断 componentType
        if component_type is None:
            component_type = self._DTYPE_TO_COMPONENT_TYPE.get(data.dtype)
            if component_type is None:
                data = data.astype(np.float32)
                component_type = self.FLOAT

        # 推断 accessor type 与 count
        if type_str is None:
            if data.ndim == 1:
                type_str = "SCALAR"
                count = len(data)
            elif data.ndim == 2:
                count = data.shape[0]
                type_str = self._COLS_TO_TYPE.get(data.shape[1], "SCALAR")
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
        else:
            count = len(data) if data.ndim == 1 else data.shape[0]

        buffer_view_idx = self._add_buffer_view(data.tobytes())

        accessor: Dict[str, Any] = {
            "bufferView":    buffer_view_idx,
            "componentType": component_type,
            "count":         count,
            "type":          type_str,
        }
        if normalized:
            accessor["normalized"] = True

        idx = len(self.accessors)
        self.accessors.append(accessor)
        return idx

    def add_image(self, image_bytes: bytes, mime_type: str, width: int, height: int) -> int:
        """
        将编码后的图像（JPEG/PNG/WebP）写入 GLB，返回 image 索引。
        """
        buffer_view_idx = self._add_buffer_view(image_bytes)
        idx = len(self.images)
        self.images.append({
            "bufferView": buffer_view_idx,
            "mimeType":   mime_type,
            "width":      width,
            "height":     height,
        })
        return idx

    def encode(self, nuviz_data: Dict[str, Any]) -> bytes:
        """
        将所有已添加的数据连同 nuviz_data 编码为完整的 GLB 文件字节。

        nuviz_data 将写入 JSON chunk 的顶层 "nuviz" 字段。
        """
        # 构建 JSON chunk
        json_obj: Dict[str, Any] = {}
        if self.buffer_views:
            json_obj["bufferViews"] = self.buffer_views
        if self.accessors:
            json_obj["accessors"] = self.accessors
        if self.images:
            json_obj["images"] = self.images
        json_obj["nuviz"] = nuviz_data

        json_bytes = json.dumps(json_obj, separators=(',', ':')).encode('utf-8')

        # 4 字节对齐（JSON 用空格填充，BIN 用零填充）
        def pad(data: bytes, fill: bytes) -> bytes:
            remainder = len(data) % 4
            return data + fill * ((4 - remainder) % 4)

        json_padded = pad(json_bytes, b' ')
        bin_padded  = pad(bytes(self.bin_data), b'\x00')

        total_length = 12 + 8 + len(json_padded) + 8 + len(bin_padded)

        glb_header        = struct.pack('<III', self.GLB_MAGIC, self.GLB_VERSION, total_length)
        json_chunk_header = struct.pack('<II',  len(json_padded), self.JSON_CHUNK_TYPE)
        bin_chunk_header  = struct.pack('<II',  len(bin_padded),  self.BIN_CHUNK_TYPE)

        return glb_header + json_chunk_header + json_padded + bin_chunk_header + bin_padded
