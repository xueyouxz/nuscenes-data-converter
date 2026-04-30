# nusviz 数据格式说明文档

> 本文档详细描述 `nusviz` 模块生成的数据结构，包括文件布局、各数据模态的存储方式、坐标系规范及解析方法。

---

## 目录

1. [概述](#1-概述)
2. [输出目录结构](#2-输出目录结构)
3. [文件格式：GLB 容器](#3-文件格式glb-容器)
4. [message_index.json](#4-message_indexjson)
5. [metadata.glb — 场景元数据](#5-metadataglb--场景元数据)
6. [messages/XXXXXX.glb — 帧状态消息](#6-messagesxxxxxxglb--帧状态消息)
7. [坐标系规范](#7-坐标系规范)
8. [类别 ID 映射](#8-类别-id-映射)
9. [场景矢量地图](#9-场景矢量地图)
10. [如何从数据中提取各类模态](#10-如何从数据中提取各类模态)
11. [GLB 二进制布局详解](#11-glb-二进制布局详解)
12. [模块文件索引](#12-模块文件索引)

---

## 1. 概述

`nusviz` 是一个将 [nuScenes](https://www.nuscenes.org/) 数据集转换为 **nuviz 格式**的工具模块。

转换结果以 **GLB（glTF 2.0 Binary）** 为容器格式，将以下多模态数据打包进统一的二进制文件：

| 数据模态 | 存储位置 | 坐标系 |
|---|---|---|
| 自车位姿（Ego Pose） | `messages/*.glb` → `poses./ego_pose` | 世界坐标系 |
| LiDAR 点云 | `messages/*.glb` → `primitives./lidar` | 世界坐标系 |
| 3D 目标检测框 | `messages/*.glb` → `primitives./objects/bounds` | 世界坐标系 |
| 相机图像（×6） | `messages/*.glb` → `primitives./camera/<CHANNEL>` | Ego 坐标系（附内外参） |
| 自车未来轨迹 | `messages/*.glb` → `primitives./ego/fut_trajectory` | 世界坐标系 |
| 对象未来轨迹 | `messages/*.glb` → `primitives./objects/fut_trajectories` | 世界坐标系 |
| 相机内外参 | `metadata.glb` → `data.cameras` | — |
| 矢量地图（×8 图层） | `metadata.glb` → `data.map` | 世界坐标系 |
| 场景元信息 | `metadata.glb` → `data.extensions.nuscenes` | — |

每个 nuScenes **sample**（约 2 Hz 采样）对应一个 `messages/XXXXXX.glb` 文件。

---

## 2. 输出目录结构

```
output/
└── <scene_name>/                  # 例如 scene-0916
    ├── metadata.glb               # 场景级别的静态元数据（一次性，不随帧变化）
    ├── message_index.json         # 所有帧的索引：时间戳 + 文件路径
    └── messages/
        ├── 000000.glb             # 第 0 帧（COMPLETE_STATE，完整初始状态）
        ├── 000001.glb             # 第 1 帧（INCREMENTAL，增量更新）
        ├── 000002.glb
        └── ...                    # 以此类推，共 N 帧（N = 场景 sample 总数）
```

### 文件角色说明

| 文件 | 类型 | 作用 |
|---|---|---|
| `metadata.glb` | 静态 | 相机内外参、场景描述、地图范围、类别映射等 |
| `message_index.json` | 索引 | 每帧的时间戳与对应文件路径的映射表 |
| `messages/000000.glb` | 动态帧（完整） | 第一帧，包含完整的初始状态 |
| `messages/XXXXXX.glb` | 动态帧（增量） | 后续帧，仅包含当帧数据 |

---

## 3. 文件格式：GLB 容器

所有 `.glb` 文件遵循 **glTF 2.0 Binary** 格式规范，由三部分组成：

```
┌─────────────────────────────────────────┐
│         GLB Header (12 bytes)           │
│  magic=0x46546C67 | version=2 | length  │
├─────────────────────────────────────────┤
│         JSON Chunk                      │
│  chunkLength | chunkType=0x4E4F534A     │
│  ┌───────────────────────────────────┐  │
│  │ { "bufferViews": [...],           │  │
│  │   "accessors":  [...],            │  │
│  │   "images":     [...],            │  │
│  │   "nuviz":      { ... }  ← 核心  │  │
│  │ }                                 │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│         BIN Chunk                       │
│  chunkLength | chunkType=0x004E4942     │
│  [ 点云数据 | 标注数据 | 图像数据 ... ]  │
└─────────────────────────────────────────┘
```

### 关键字段

- **`bufferViews`**：描述 BIN chunk 中各数据段的字节偏移量（`byteOffset`）和长度（`byteLength`），4 字节对齐。
- **`accessors`**：描述如何解释 bufferView 中的二进制数据，包含 `componentType`（数据类型，如 `FLOAT=5126`、`UNSIGNED_INT=5125`）和 `type`（数据维度，如 `VEC3`、`SCALAR`）。
- **`images`**：存储图像文件的 bufferView 引用和 MIME 类型。
- **`nuviz`**：**nuviz 专有字段**，包含所有语义数据（位姿、点云引用、标注框引用、图像引用等）。

### accessor 引用语法

nuviz 字段内通过 JSON Pointer 引用 accessor 和 image：

```
"#/accessors/0"   →  accessors 数组第 0 个元素
"#/images/2"      →  images 数组第 2 个元素
```

### BIN chunk 对齐规则

- 每段数据写入前按 4 字节对齐，用 `\x00` 填充。
- JSON chunk 末尾用空格（`0x20`）填充至 4 字节对齐。
- BIN chunk 末尾用零字节填充至 4 字节对齐。

---

## 4. message_index.json

`message_index.json` 是场景级别的帧索引文件，消费方通过它按时间顺序加载每一帧。

### 完整结构

```json
{
  "message_format": "BINARY",
  "metadata": "metadata.glb",
  "log_info": {
    "start_time": 1533151709.572,
    "end_time":   1533151729.872
  },
  "messages": [
    { "index": 0,  "timestamp": 1533151709.572, "file": "messages/000000.glb" },
    { "index": 1,  "timestamp": 1533151711.072, "file": "messages/000001.glb" },
    ...
    { "index": 40, "timestamp": 1533151729.872, "file": "messages/000040.glb" }
  ],
  "extensions": {
    "nuscenes": {
      "scene_token": "cc8c0bf57f984915a77078b10eb33198",
      "scene_name":  "scene-0916",
      "mapId":       "singapore-onenorth"
    }
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `message_format` | string | 固定值 `"BINARY"`，表示消息为二进制 GLB 格式 |
| `metadata` | string | metadata.glb 的相对路径 |
| `log_info.start_time` | float | 场景开始时间，Unix 时间戳（秒），由 nuScenes timestamp（微秒）除以 1e6 得到 |
| `log_info.end_time` | float | 场景结束时间（秒） |
| `messages[].index` | int | 帧序号，从 0 开始 |
| `messages[].timestamp` | float | 该帧时间戳（秒） |
| `messages[].file` | string | 该帧 GLB 文件的相对路径 |
| `extensions.nuscenes.scene_token` | string | nuScenes 原始 scene token |
| `extensions.nuscenes.mapId` | string | 地图位置标识，用于加载地图瓦片，取自 `log.location` |

> **采样频率**：nuScenes 约 2 Hz，相邻帧时间差约 0.5 秒。

---

## 5. metadata.glb — 场景元数据

`metadata.glb` 在场景开始时加载一次，包含整个场景不变的静态信息。

其 JSON chunk 中 `nuviz` 字段的完整结构如下：

```json
{
  "type": "nuviz/metadata",
  "data": {
    "log_info": {
      "start_time": 1533151709.572,
      "end_time":   1533151729.872
    },
    "streams": {
      "/ego_pose":       { "category": "POSE",      "coordinate": "world" },
      "/lidar":          { "category": "PRIMITIVE", "type": "point",           "coordinate": "world" },
      "/objects/bounds": { "category": "PRIMITIVE", "type": "nuscenes_cuboid", "coordinate": "world" },
      "/camera/CAM_FRONT":       { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/camera/CAM_FRONT_LEFT":  { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/camera/CAM_FRONT_RIGHT": { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/camera/CAM_BACK":        { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/camera/CAM_BACK_LEFT":   { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/camera/CAM_BACK_RIGHT":  { "category": "PRIMITIVE", "type": "image", "coordinate": "ego" },
      "/map":                    { "category": "PRIMITIVE", "type": "map_geometry", "coordinate": "world" }
    },
    "cameras": {
      "CAM_FRONT": {
        "image_width":  1600,
        "image_height": 900,
        "intrinsic": [
          [1266.417, 0.0,      816.267],
          [0.0,      1266.417, 491.507],
          [0.0,      0.0,      1.0    ]
        ],
        "extrinsic": {
          "translation": [1.72200568, 0.00475453, 1.49491292],
          "rotation":    [0.9999, 0.0000, 0.0071, 0.0105]
        }
      },
      "CAM_FRONT_LEFT":  { ... },
      "CAM_FRONT_RIGHT": { ... },
      "CAM_BACK":        { ... },
      "CAM_BACK_LEFT":   { ... },
      "CAM_BACK_RIGHT":  { ... }
    },
    "map": {
      "buffer_radius_m": 75,
      "layers": {
        "drivable_area": { "vertices": "#/accessors/12", "counts": "#/accessors/13" },
        "road_segment":  { "vertices": "#/accessors/14", "counts": "#/accessors/15" },
        "lane":          { "vertices": "#/accessors/16", "counts": "#/accessors/17" },
        "lane_connector":{ "vertices": "#/accessors/18", "counts": "#/accessors/19" },
        "ped_crossing":  { "vertices": "#/accessors/20", "counts": "#/accessors/21" },
        "walkway":       { "vertices": "#/accessors/22", "counts": "#/accessors/23" },
        "stop_line":     { "vertices": "#/accessors/24", "counts": "#/accessors/25" },
        "carpark_area":  { "vertices": "#/accessors/26", "counts": "#/accessors/27" }
      }
    },
    "extensions": {
      "nuscenes": {
        "scene": {
          "scene_token": "cc8c0bf57f984915a77078b10eb33198",
          "name":        "scene-0916",
          "description": "Parked truck, construction, one-way",
          "location":    "singapore-onenorth",
          "mapId":       "singapore-onenorth"
        },
        "map": {
          "canvas_edge_m": [-300.0, -1500.0, 1500.0, 1000.0]
        },
        "coordinate": {
          "units":            "meter",
          "matrixConvention": "nuscenes",
          "quatOrder":        "wxyz"
        },
        "mapping": {
          "classes": {
            "nameToId": {
              "barrier": 1, "bicycle": 2, "bus": 3, "car": 4,
              "construction_vehicle": 5, "motorcycle": 6, "pedestrian": 7,
              "traffic_cone": 8, "trailer": 9, "truck": 10,
              "driveable_surface": 11, "other_flat": 12, "sidewalk": 13,
              "terrain": 14, "manmade": 15, "vegetation": 16
            }
          }
        }
      }
    }
  }
}
```

### 关键子结构说明

#### `streams`

声明场景包含哪些数据通道（stream），以及每个通道的语义类型和所在坐标系。Stream 路径与 `messages/*.glb` 中 `primitives` 和 `poses` 的 key 严格对应。

| Stream 路径 | category | type | coordinate | 说明 |
|---|---|---|---|---|
| `/ego_pose` | POSE | — | world | 自车在世界坐标系中的位姿 |
| `/lidar` | PRIMITIVE | point | world | 世界坐标系下的 LiDAR 点云 |
| `/objects/bounds` | PRIMITIVE | nuscenes_cuboid | world | 3D 检测框，世界坐标系 |
| `/camera/<CHANNEL>` | PRIMITIVE | image | ego | 相机图像，附相机外参（来自 calibrated_sensor，描述 sensor→ego 方向） |
| `/map` | PRIMITIVE | map_geometry | world | 矢量地图图层多边形，存于 `metadata.glb`，不随帧变化 |
| `/ego/fut_trajectory` | PRIMITIVE | ego_trajectory | world | 自车从当前帧到末帧的未来位置序列 |
| `/objects/fut_trajectories` | PRIMITIVE | object_trajectories | world | 当前帧所有对象的未来中心点序列（CSR 格式） |

#### `cameras`

每个相机通道的内参和外参（**在整个场景中保持不变**）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_width` | int | 图像宽度（像素），nuScenes 标准为 1600 |
| `image_height` | int | 图像高度（像素），nuScenes 标准为 900 |
| `intrinsic` | float[3][3] | 3×3 相机内参矩阵 K，来自 `calibrated_sensor.camera_intrinsic` |
| `extrinsic.translation` | float[3] | 相机在 ego 坐标系中的平移 [x, y, z]（米） |
| `extrinsic.rotation` | float[4] | 相机在 ego 坐标系中的旋转，四元数 [w, x, y, z] |

> **注意**：`extrinsic` 描述的是**相机在 ego 坐标系中的位置和朝向**，来源于 nuScenes `calibrated_sensor` 表。
>
> **投影方向注意**：nuScenes `calibrated_sensor.rotation` 描述的是从**传感器坐标系**到 **ego 坐标系**的旋转（`sensor → ego`）。在投影链中需要将其**转置/取逆**得到 `ego → sensor` 的旋转。因此前端投影时应使用 `(T_cam_ego)⁻¹ × (T_ego_world)⁻¹` 而非直接相乘。

#### `map.canvas_edge_m`

地图画布的世界坐标范围 `[xmin, ymin, xmax, ymax]`（单位：米）：

| 地图位置 | xmin | ymin | xmax | ymax |
|---|---|---|---|---|
| singapore-onenorth | -300 | -1500 | 1500 | 1000 |
| singapore-hollandvillage | -300 | -1500 | 1500 | 1000 |
| singapore-queenstown | -300 | -1500 | 1500 | 1000 |
| boston-seaport | -2000 | -1000 | 2000 | 2000 |
| （默认） | -2000 | -2000 | 2000 | 2000 |

---

## 6. messages/XXXXXX.glb — 帧状态消息

每一帧的 GLB 文件中，`nuviz` 字段携带一条 `nuviz/state_update` 消息。

### 顶层结构

```json
{
  "type": "nuviz/state_update",
  "data": {
    "update_type": "COMPLETE_STATE",
    "updates": [
      {
        "timestamp": 1533151709.572,
        "poses": {
          "/ego_pose": { "translation": [...], "rotation": [...] }
        },
        "primitives": {
          "/lidar":          { "points":   [...] },
          "/objects/bounds": { "cuboids":  [...] },
          "/camera/CAM_FRONT":      { "images": [...] },
          "/camera/CAM_FRONT_LEFT": { "images": [...] },
          ...
        }
      }
    ]
  }
}
```

### `update_type` 字段

| 值 | 含义 | 出现时机 |
|---|---|---|
| `COMPLETE_STATE` | 完整状态帧，包含场景所有数据 | 仅第 0 帧（`000000.glb`） |
| `INCREMENTAL` | 增量帧，仅包含当帧新数据 | 第 1 帧起所有后续帧 |

### `updates[0].timestamp`

当前帧的 Unix 时间戳（秒），由 nuScenes `sample.timestamp`（微秒）除以 1e6 得到。

---

### 6.1 Ego Pose（自车位姿）

路径：`data.updates[0].poses."/ego_pose"`

```json
{
  "translation": [314.267, 1231.937, 0.753],
  "rotation":    [0.9999, 0.0003, 0.0016, -0.0100]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `translation` | float[3] | 自车在世界坐标系中的位置 [x, y, z]，单位：米 |
| `rotation` | float[4] | 自车在世界坐标系中的朝向，四元数 [w, x, y, z] |

**数据来源**：通过 `LIDAR_TOP` 对应的 `ego_pose` 记录读取（`sample_data.ego_pose_token`）。

**跨帧轨迹**：将所有帧的 `ego_pose.translation` 按时间顺序连接，即得到完整的**自车行驶轨迹**。

---

### 6.2 LiDAR 点云

路径：`data.updates[0].primitives."/lidar"`

```json
{
  "points": [
    {
      "points": "#/accessors/0",
      "extensions": {
        "nuscenes": {
          "INTENSITY": "#/accessors/1"
        }
      }
    }
  ]
}
```

#### accessor 数据格式

| accessor 引用 | glTF type | componentType | dtype | shape | 说明 |
|---|---|---|---|---|---|
| `points` | VEC3 | FLOAT (5126) | float32 | (N, 3) | 世界坐标系下点云 XYZ，单位：米 |
| `INTENSITY` | SCALAR | FLOAT (5126) | float32 | (N,) | 激光反射强度，范围约 [0, 255] |

**原始数据格式**：nuScenes LiDAR 文件（`.pcd.bin`）为 float32 的二进制数组，每点 5 个 float32：

```
[x, y, z, intensity, ring_index]
```

**坐标变换**：点云经过两级变换从传感器坐标系变换到世界坐标系：

```
传感器坐标系 → ego 坐标系 → 世界坐标系

p_ego   = R_sensor * p_sensor + t_sensor
p_world = R_ego    * p_ego    + t_ego
```

其中 `R_sensor`, `t_sensor` 来自 `calibrated_sensor` 表，`R_ego`, `t_ego` 来自 `ego_pose` 表。

---

### 6.3 目标检测框（3D Cuboids）

路径：`data.updates[0].primitives."/objects/bounds"`

```json
{
  "cuboids": [
    {
      "count":    42,
      "CENTER":   "#/accessors/2",
      "SIZE":     "#/accessors/3",
      "ROTATION": "#/accessors/4",
      "CLASS_ID": "#/accessors/5",
      "TRACK_ID": "#/accessors/6"
    }
  ]
}
```

#### accessor 数据格式

| 字段 | glTF type | componentType | dtype | shape | 说明 |
|---|---|---|---|---|---|
| `CENTER` | VEC3 | FLOAT (5126) | float32 | (M, 3) | 目标框中心 [x, y, z]，世界坐标系，单位：米 |
| `SIZE` | VEC3 | FLOAT (5126) | float32 | (M, 3) | 目标框尺寸 [width, length, height]，单位：米 |
| `ROTATION` | VEC4 | FLOAT (5126) | float32 | (M, 4) | 目标朝向四元数 [w, x, y, z]，世界坐标系 |
| `CLASS_ID` | SCALAR | UNSIGNED_INT (5125) | uint32 | (M,) | 类别 ID（见[第 8 节](#8-类别-id-映射)），未知类别为 0 |
| `TRACK_ID` | SCALAR | UNSIGNED_INT (5125) | uint32 | (M,) | 跨帧追踪 ID，来自 nuScenes `instance` 表的索引 |

M 为当前帧的目标总数（即 `sample.anns` 列表长度）。

**坐标说明**：nuScenes `sample_annotation.translation` 和 `rotation` 已经是**世界坐标系**，无需额外变换，直接存储。

**跨帧追踪**：同一个物理目标在不同帧中的 `TRACK_ID` 保持不变，可用于构建车辆/行人轨迹。`TRACK_ID` 值等于该目标 `instance_token` 在 nuScenes `instance` 表中的整数索引（`nusc.getind('instance', ann['instance_token'])`）。

**尺寸约定**：`SIZE = [width, length, height]`，对应 nuScenes `annotation.size` 的原始顺序（wlh）。

---

### 6.4 相机图像

路径：`data.updates[0].primitives."/camera/<CHANNEL>"`

六个相机通道：`CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`

```json
{
  "images": [
    {
      "data": "#/images/0"
    }
  ]
}
```

`#/images/0` 引用 GLB JSON chunk 中 `images` 数组的第 0 个元素：

```json
{
  "bufferView": 7,
  "mimeType":   "image/jpeg",
  "width":      1600,
  "height":     900
}
```

实际图像字节数据存储于 BIN chunk 对应的 bufferView 中。

| 字段 | 值 | 说明 |
|---|---|---|
| `mimeType` | `"image/jpeg"` | JPEG 压缩，quality=85 |
| `width` | 1600 | 图像宽度（像素） |
| `height` | 900 | 图像高度（像素） |

**获取相机外参**：相机图像本身不嵌入外参，外参统一存储在 `metadata.glb` 的 `data.cameras.<CHANNEL>.extrinsic` 中，读取一次即可在整个场景复用。

---

### 6.5 自车未来轨迹

路径：`data.updates[0].primitives."/ego/fut_trajectory"`

```json
{
  "trajectory": [
    {
      "poses": "#/accessors/N",
      "count": 32
    }
  ]
}
```

#### accessor 数据格式

| accessor 引用 | glTF type | componentType | dtype | shape | 说明 |
|---|---|---|---|---|---|
| `poses` | VEC3 | FLOAT (5126) | float32 | (M, 3) | 从当前帧（含）到末帧，自车在世界坐标系中的位置序列 `[x, y, z]`，单位：米；M = 剩余帧数（含当前帧） |

**语义**：`count` 等于 `M`，即轨迹点总数。第 K 帧的 `poses` 数组包含第 K 帧到第 N-1 帧（N = 总帧数）共 N-K 个位置点，index 0 对应当前帧自身位置。

**仅存储 translation**：不含朝向信息，前端连线绘制路径即可，无需旋转数据。

**坐标系**：与 `poses./ego_pose` 一致，均为世界坐标系（X 东、Y 北、Z 上）。

**写入规则**：每帧均写入此流，由 `converter.py` 在转换场景前预计算全场景 ego 位置列表，第 K 帧取其 `[K:]` 切片传入 `MessageBuilder`。

---

### 6.6 对象未来轨迹

路径：`data.updates[0].primitives."/objects/fut_trajectories"`

```json
{
  "trajectories": [
    {
      "points":    "#/accessors/N",
      "offsets":   "#/accessors/N+1",
      "obj_count": 42
    }
  ]
}
```

#### accessor 数据格式

| accessor 引用 | glTF type | componentType | dtype | shape | 说明 |
|---|---|---|---|---|---|
| `points` | VEC3 | FLOAT (5126) | float32 | (T, 3) | 所有对象所有未来轨迹点的中心位置 `[x, y, z]`，世界坐标系，单位：米；T = 所有对象未来点数之和 |
| `offsets` | SCALAR | UNSIGNED_INT (5125) | uint32 | (M+1,) | CSR 格式起始偏移数组；第 i 个对象的轨迹点为 `points[offsets[i] : offsets[i+1]]`；`M` = `obj_count`，`offsets[M]` = `T` |

**CSR 分段示例**（3 个对象，轨迹点数分别为 5、3、7）：

```
offsets = [0, 5, 8, 15]   # 长度 = obj_count + 1 = 4
points  = [p0_0..p0_4,    # 对象 0 的 5 个点
           p1_0..p1_2,    # 对象 1 的 3 个点
           p2_0..p2_6]    # 对象 2 的 7 个点
```

**对象顺序**：与同帧 `/objects/bounds` 中 `cuboids[0]` 的对象顺序**严格一一对应**，即第 i 条轨迹对应第 i 个 cuboid（相同的 `TRACK_ID`）。

**缺帧处理**：若某对象在当前帧之后某帧不可见，该帧在此对象轨迹中跳过，不插入占位点；仅记录实际出现的关键帧位置。

**坐标系**：与 `/objects/bounds` 的 `CENTER` 字段一致，均为世界坐标系。

**写入规则**：由 `converter.py` 预计算全场景 `track_id → [(frame_idx, [x,y,z]), ...]` 映射，每帧按当前 `anns` 顺序过滤 `frame_idx >= current_frame_idx` 的点传入 `MessageBuilder`。

---

## 7. 坐标系规范

### 7.1 坐标系定义

nusviz 使用三个坐标系：

#### 世界坐标系（World Frame）

- **原点**：nuScenes 地图的绝对坐标原点（UTM 投影）
- **单位**：米
- **轴向**：X 向东，Y 向北，Z 向上（右手系）
- **包含数据**：ego_pose、LiDAR 点云、3D 目标框

#### Ego 坐标系（Ego Frame / 车体坐标系）

- **原点**：自车车体几何中心（随车运动）
- **单位**：米
- **轴向**：X 向前，Y 向左，Z 向上（右手系）
- **包含数据**：相机外参的参考系
- **与世界坐标系的关系**：由 `ego_pose.translation` 和 `ego_pose.rotation` 定义

#### 传感器坐标系（Sensor Frame）

- **原点**：各传感器安装位置
- **单位**：米
- **包含数据**：LiDAR 原始点云读取后在此坐标系，经变换后转为世界坐标系
- **与 ego 坐标系的关系**：由 `calibrated_sensor.translation` 和 `calibrated_sensor.rotation` 定义

### 7.2 变换链

```
传感器坐标系
    │  R_sensor, t_sensor（来自 calibrated_sensor 表）
    ▼
ego 坐标系
    │  R_ego, t_ego（来自 ego_pose 表，随帧变化）
    ▼
世界坐标系
```

**点变换公式**（以 LiDAR 点云为例）：

```
p_ego   = R_sensor × p_sensor + t_sensor
p_world = R_ego    × p_ego    + t_ego
```

展开为单步变换：

```
p_world = R_ego × (R_sensor × p_sensor + t_sensor) + t_ego
        = (R_ego × R_sensor) × p_sensor + (R_ego × t_sensor + t_ego)
```

### 7.3 四元数约定

**顺序**：`[w, x, y, z]`（wxyz 顺序）

这与 nuScenes 原始数据的存储顺序（`[w, x, y, z]`）一致，与 `pyquaternion.Quaternion` 的构造顺序一致。

```python
# nuScenes raw: [w, x, y, z]
q = Quaternion(ego_pose['rotation'])   # pyquaternion 接受 [w, x, y, z]

# 转换为 nuviz 存储格式
rotation_wxyz = [q.w, q.x, q.y, q.z]  # 仍然是 [w, x, y, z]
```

> **注意**：部分框架（如 ROS、scipy）使用 `[x, y, z, w]` 顺序，读取时需注意转换。

### 7.4 旋转方向与投影变换

nuScenes 存储的旋转描述的是**正向**（从局部坐标系到父坐标系）变换：

| 数据来源 | rotation 描述的变换方向 |
|---|---|
| `ego_pose.rotation` | ego → world |
| `calibrated_sensor.rotation` | sensor → ego |

投影计算需要**逆变换**（旋转矩阵正交，转置等于逆）：

```
T_world_ego = (T_ego_world)⁻¹ = T_ego_world.T   # 世界 → Ego
T_ego_cam  = (T_cam_ego)⁻¹  = T_cam_ego.T     # Ego → 相机
```

完整投影链（世界 3D 点 → 图像像素）：

```
p_image ~ K × T_ego_cam × T_world_ego × p_world
       = K × (T_cam_ego)⁻¹ × (T_ego_world)⁻¹ × p_world
```

---

## 8. 类别 ID 映射

### 检测类别（用于 `CLASS_ID` accessor）

| ID | 简化名 | nuScenes 原始类别前缀 |
|---|---|---|
| 1 | barrier | barrier |
| 2 | bicycle | vehicle.bicycle |
| 3 | bus | vehicle.bus |
| 4 | car | vehicle.car |
| 5 | construction_vehicle | vehicle.construction |
| 6 | motorcycle | vehicle.motorcycle |
| 7 | pedestrian | pedestrian |
| 8 | traffic_cone | traffic_cone |
| 9 | trailer | vehicle.trailer |
| 10 | truck | vehicle.truck |

> **未知类别**：若 `category_name` 不匹配上述前缀，`CLASS_ID` 记为 `0`。

### 语义分割类别（用于 metadata 的 `mapping.classes`，暂未写入点云）

| ID | 名称 | 说明 |
|---|---|---|
| 11 | driveable_surface | 可行驶路面 |
| 12 | other_flat | 其他平坦地面 |
| 13 | sidewalk | 人行道 |
| 14 | terrain | 地形 |
| 15 | manmade | 人造建筑 |
| 16 | vegetation | 植被 |

### 类别名解析逻辑

nuScenes 原始类别名为层级结构（如 `vehicle.car.sedan`），解析规则如下：

```python
# 按 _CATEGORY_MAP 中定义的前缀顺序依次匹配
for prefix, simple_name in _CATEGORY_MAP.items():
    if prefix in category_name:   # 前缀包含检查
        return simple_name
# 回退：取第一级（如 'animal.dog' -> 'animal'）
return category_name.split('.')[0]
```

---

## 9. 场景矢量地图

矢量地图是 `metadata.glb` 新增的静态数据，场景初始化时加载一次，整个场景复用，不写入任何帧消息（`messages/*.glb`）。

### 9.1 地图数据结构

地图数据存放在 `nuviz.data.map` 字段中，包含两个顶层字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `buffer_radius_m` | float | 生成时以自车轨迹为中心线、向两侧延伸的缓冲半径（单位：米），默认 75 |
| `layers` | object | 按图层名索引，每个图层包含 `vertices` 和 `counts` 两个 accessor 引用 |

每个图层用两个 accessor 表示：

| accessor 字段 | glTF type | componentType | dtype | shape | 说明 |
|---|---|---|---|---|---|
| `vertices` | VEC3 | FLOAT (5126) | float32 | (K, 3) | 该图层所有多边形顶点按序拼接，`[x, y, 0.0]`，世界坐标系，单位米 |
| `counts` | SCALAR | UNSIGNED_INT (5125) | uint32 | (P,) | 每个多边形的顶点数，`counts[i]` 为第 i 个多边形的顶点数量 |

消费方通过 `counts` 做前缀求和，从 `vertices` 中切分出每个独立多边形：

```
多边形 0 的顶点：vertices[ 0 : counts[0] ]
多边形 1 的顶点：vertices[ counts[0] : counts[0]+counts[1] ]
多边形 i 的顶点：vertices[ sum(counts[:i]) : sum(counts[:i+1]) ]
```

### 9.2 支持的地图图层

| 图层名 | 语义 | 典型用途 |
|---|---|---|
| `drivable_area` | 可行驶区域 | 渲染可行驶区域底图 |
| `road_segment` | 道路段 | 渲染道路边界 |
| `lane` | 车道 | 车道划分线、车道级规划分析 |
| `lane_connector` | 车道连接段（交叉口） | 渲染交叉口通行关系 |
| `ped_crossing` | 人行横道 | 渲染斑马线 |
| `walkway` | 人行道 | 渲染人行道 |
| `stop_line` | 停止线 | 渲染停止线 |
| `carpark_area` | 停车场 | 渲染停车场区域 |

> 仅写入精筛后存在几何数据的图层，空图层不出现在 `layers` 中。

### 9.3 地图范围确定方法

地图不提取整张 nuScenes 地图，而是沿自车行驶轨迹向两侧延伸 `buffer_radius_m`（默认 75 m）构建缓冲区，仅保留缓冲区内的地图元素：

```
① 遍历场景所有帧，收集 LIDAR_TOP ego_pose 的 [x, y] 轨迹点
② 用 Shapely 将轨迹点构造为 LineString，调用 .buffer(75) 得到「跑道形」多边形
③ nusc_map.get_records_in_patch(bbox)   ← 矩形粗筛
④ buffer_poly.intersects(geom)          ← 精筛（Shapely 几何相交）
⑤ buffer_poly.intersection(geom)        ← 裁剪（仅保留缓冲区内的几何部分）
```

缓冲半径 75 m 可覆盖自车两侧约 21 条车道宽度，足以包含完整道路结构及路口附近的地图元素。

### 9.4 BIN chunk 中地图数据的布局

地图 accessor 写入 `metadata.glb` 的 BIN chunk，按图层顺序依次排列：

```
BIN chunk（metadata.glb）
├── [bufferView  0]  drivable_area  vertices   float32 (K0, 3)
├── [padding]        4 字节对齐
├── [bufferView  1]  drivable_area  counts     uint32  (P0,)
├── [padding]
├── [bufferView  2]  road_segment   vertices   float32 (K1, 3)
├── [padding]
├── [bufferView  3]  road_segment   counts     uint32  (P1,)
├── [padding]
├── ...
└── [bufferView 15]  carpark_area   counts     uint32  (P7,)
```

### 9.5 地图可视化工具

`visualize_map.py` 从已生成的 `metadata.glb` 中读取矢量地图并渲染为 PNG 图像。

**用法：**

```bash
# 渲染指定场景
python visualize_map.py output/scene-0916

# 渲染 output/ 下所有场景
python visualize_map.py
```

**输出：** 保存为 `output/<scene_name>/map_<scene_name>.png`，包含各图层多边形及自车行驶轨迹叠加显示。

**图层配色：**

| 图层 | 填充色 | 边框色 |
|---|---|---|
| `drivable_area` | `#C8D8E8`（浅蓝灰） | `#7A9AB5` |
| `road_segment` | `#D6D6D6`（浅灰） | `#888888` |
| `lane` | `#E8E0C8`（米黄） | `#B8A878` |
| `lane_connector` | `#F0D8A0`（浅黄） | `#C8A840` |
| `ped_crossing` | `#F0C8C8`（浅红） | `#D07070` |
| `walkway` | `#C8E8C8`（浅绿） | `#70A870` |
| `stop_line` | `#F08080`（红） | `#C83030` |
| `carpark_area` | `#E0C8F0`（浅紫） | `#9060C0` |

自车轨迹以橙色线（`#FF6B35`）叠加绘制，起点标绿色圆点，终点标红色方块。

---

## 10. 如何从数据中提取各类模态

以下示例均以 Python 为基础，展示如何从已生成的 GLB 文件中读取各类数据。

### 前置：解析 GLB 文件

```python
import struct, json

def parse_glb(path):
    """解析 GLB 文件，返回 (json_data, bin_data)。"""
    with open(path, 'rb') as f:
        magic, version, total_len = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, "Not a GLB file"

        # JSON chunk
        json_len, json_type = struct.unpack('<II', f.read(8))
        json_data = json.loads(f.read(json_len).decode('utf-8'))

        # BIN chunk
        bin_len, bin_type = struct.unpack('<II', f.read(8))
        bin_data = f.read(bin_len)

    return json_data, bin_data

def read_accessor(json_data, bin_data, accessor_ref):
    """通过 accessor 引用（如 '#/accessors/0'）读取 numpy 数组。"""
    import numpy as np
    idx = int(accessor_ref.split('/')[-1])
    acc = json_data['accessors'][idx]
    bv  = json_data['bufferViews'][acc['bufferView']]

    dtype_map = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    cols_map  = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

    dtype = dtype_map[acc['componentType']]
    cols  = cols_map[acc['type']]
    count = acc['count']

    raw = bin_data[bv['byteOffset'] : bv['byteOffset'] + bv['byteLength']]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(count, cols) if cols > 1 else arr
```

---

### 10.1 提取自车轨迹

遍历所有帧，读取 `ego_pose`，构建轨迹点序列。

```python
import json
from pathlib import Path


def extract_ego_trajectory(scene_dir):
    """
    返回自车轨迹列表：
    [{'timestamp': float, 'translation': [x,y,z], 'rotation': [w,x,y,z]}, ...]
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    trajectory = []

    for entry in index['messages']:
        json_data, bin_data = parse_glb(scene_dir / entry['file'])
        nuviz = json_data['nuviz']
        update = nuviz['data']['updates'][0]
        pose = update['poses']['/ego_pose']
        trajectory.append({
            'timestamp': update['timestamp'],
            'translation': pose['translation'],  # [x, y, z] 世界坐标系（米）
            'rotation': pose['rotation'],  # [w, x, y, z]
        })

    return trajectory


# 示例：提取 XY 平面轨迹
traj = extract_ego_trajectory('../nusviz/output/scene-0916')
xs = [p['translation'][0] for p in traj]
ys = [p['translation'][1] for p in traj]
```

---

### 10.2 提取相机图像

相机内外参从 `metadata.glb` 读取，图像数据从各帧 `messages/*.glb` 读取。

```python
import io
from PIL import Image
import numpy as np

def load_camera_params(scene_dir):
    """从 metadata.glb 读取所有相机的内外参。"""
    scene_dir = Path(scene_dir)
    json_data, _ = parse_glb(scene_dir / 'metadata.glb')
    return json_data['nuviz']['data']['cameras']
    # 返回格式：{'CAM_FRONT': {'image_width':1600, 'image_height':900,
    #                         'intrinsic':[[...],[...],[...]], 'extrinsic':{...}}, ...}

def extract_camera_image(scene_dir, frame_idx, channel='CAM_FRONT'):
    """
    提取指定帧、指定相机通道的图像。
    返回 PIL.Image 对象。
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    entry = index['messages'][frame_idx]

    json_data, bin_data = parse_glb(scene_dir / entry['file'])
    nuviz      = json_data['nuviz']
    primitives = nuviz['data']['updates'][0]['primitives']
    cam_prim   = primitives.get(f'/camera/{channel}')
    if cam_prim is None:
        return None

    image_ref = cam_prim['images'][0]['data']   # 如 '#/images/0'
    img_idx   = int(image_ref.split('/')[-1])
    img_meta  = json_data['images'][img_idx]
    bv        = json_data['bufferViews'][img_meta['bufferView']]

    img_bytes = bin_data[bv['byteOffset'] : bv['byteOffset'] + bv['byteLength']]
    return Image.open(io.BytesIO(img_bytes))

# 示例：读取第 5 帧前视相机图像
cam_params = load_camera_params('output/scene-0916')
K = np.array(cam_params['CAM_FRONT']['intrinsic'])   # 3x3 内参矩阵
extrinsic  = cam_params['CAM_FRONT']['extrinsic']    # translation + rotation
img = extract_camera_image('output/scene-0916', frame_idx=5, channel='CAM_FRONT')
img.save('frame5_front.jpg')
```

---

### 10.3 提取 LiDAR 点云

```python
import numpy as np

def extract_lidar(scene_dir, frame_idx):
    """
    提取指定帧的 LiDAR 点云（世界坐标系）。
    返回 (points_xyz, intensity)：
        points_xyz: (N, 3) float32，世界坐标系，单位米
        intensity:  (N,)   float32，激光反射强度
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    entry = index['messages'][frame_idx]

    json_data, bin_data = parse_glb(scene_dir / entry['file'])
    nuviz  = json_data['nuviz']
    update = nuviz['data']['updates'][0]
    lidar  = update['primitives']['/lidar']
    point_prim = lidar['points'][0]

    points_xyz = read_accessor(json_data, bin_data, point_prim['points'])
    intensity  = read_accessor(json_data, bin_data,
                               point_prim['extensions']['nuscenes']['INTENSITY'])
    return points_xyz, intensity

# 示例：读取第 0 帧点云，统计点数
xyz, intensity = extract_lidar('output/scene-0916', frame_idx=0)
print(f'点云数量: {len(xyz)}')        # 通常约 30000~40000 点
print(f'XYZ 范围: {xyz.min(0)} ~ {xyz.max(0)}')
```

---

### 10.4 提取目标标注框

```python
import numpy as np

def extract_objects(scene_dir, frame_idx):
    """
    提取指定帧的 3D 目标标注框（世界坐标系）。
    返回字典：
    {
        'centers':    (M, 3) float32  — 框中心 [x,y,z]，世界坐标系，米
        'sizes':      (M, 3) float32  — 框尺寸 [w,l,h]，米
        'rotations':  (M, 4) float32  — 旋转四元数 [w,x,y,z]，世界坐标系
        'class_ids':  (M,)   uint32   — 类别 ID（0=unknown, 1~10 见类别表）
        'track_ids':  (M,)   uint32   — 跨帧追踪 ID
    }
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    entry = index['messages'][frame_idx]

    json_data, bin_data = parse_glb(scene_dir / entry['file'])
    nuviz  = json_data['nuviz']
    update = nuviz['data']['updates'][0]
    bounds = update['primitives'].get('/objects/bounds')
    if bounds is None:
        return None

    cuboid = bounds['cuboids'][0]
    return {
        'centers':   read_accessor(json_data, bin_data, cuboid['CENTER']),
        'sizes':     read_accessor(json_data, bin_data, cuboid['SIZE']),
        'rotations': read_accessor(json_data, bin_data, cuboid['ROTATION']),
        'class_ids': read_accessor(json_data, bin_data, cuboid['CLASS_ID']),
        'track_ids': read_accessor(json_data, bin_data, cuboid['TRACK_ID']),
    }

# 示例：读取第 0 帧标注，按 TRACK_ID 构建跨帧轨迹
ID_TO_NAME = {
    1:'barrier', 2:'bicycle', 3:'bus', 4:'car', 5:'construction_vehicle',
    6:'motorcycle', 7:'pedestrian', 8:'traffic_cone', 9:'trailer', 10:'truck'
}

from collections import defaultdict
track_trajectories = defaultdict(list)

index = json.loads(open('output/scene-0916/message_index.json').read())
for entry in index['messages']:
    objs = extract_objects('output/scene-0916', entry['index'])
    if objs is None:
        continue
    for i in range(len(objs['centers'])):
        tid = int(objs['track_ids'][i])
        track_trajectories[tid].append({
            'timestamp': entry['timestamp'],
            'center':    objs['centers'][i].tolist(),
            'class_id':  int(objs['class_ids'][i]),
        })

# track_trajectories[tid] 即为 ID=tid 目标的完整运动轨迹
```

---

### 10.5 提取矢量地图

矢量地图从 `metadata.glb` 读取，加载一次即可在整个场景复用。

```python
import numpy as np
from pathlib import Path

def extract_map_layer(scene_dir, layer_name):
    """
    从 metadata.glb 中读取指定图层的矢量地图多边形。

    参数：
        scene_dir:  场景目录路径，如 'output/scene-0916'
        layer_name: 图层名，如 'drivable_area'、'lane' 等

    返回：
        polygons: list of ndarray, 每个元素为 (Ni, 3) float32，
                  表示一个多边形的顶点数组，世界坐标系 XYZ（Z=0），单位米
    """
    scene_dir = Path(scene_dir)
    json_data, bin_data = parse_glb(scene_dir / 'metadata.glb')
    map_data = json_data['nuviz']['data']['map']

    if layer_name not in map_data['layers']:
        return []

    layer    = map_data['layers'][layer_name]
    vertices = read_accessor(json_data, bin_data, layer['vertices'])  # (K, 3) float32
    counts   = read_accessor(json_data, bin_data, layer['counts'])    # (P,)   uint32

    polygons = []
    offset = 0
    for n in counts:
        polygons.append(vertices[offset : offset + n])
        offset += n
    return polygons


def load_all_map_layers(scene_dir):
    """
    一次性加载所有图层，返回字典。
    返回：{layer_name: list of (Ni, 3) ndarray}
    """
    scene_dir = Path(scene_dir)
    json_data, bin_data = parse_glb(scene_dir / 'metadata.glb')
    map_data = json_data['nuviz']['data']['map']
    result = {}
    for layer_name in map_data.get('layers', {}):
        result[layer_name] = extract_map_layer(scene_dir, layer_name)
    return result


# 示例：读取 drivable_area 图层并统计多边形数
map_layers = load_all_map_layers('output/scene-0916')
print(f"drivable_area 多边形数: {len(map_layers['drivable_area'])}")
print(f"lane 多边形数:          {len(map_layers['lane'])}")

# 示例：获取所有 drivable_area 顶点的 XY 范围
all_verts = np.concatenate(map_layers['drivable_area'], axis=0)  # (K_total, 3)
print(f"X 范围: {all_verts[:, 0].min():.1f} ~ {all_verts[:, 0].max():.1f} m")
print(f"Y 范围: {all_verts[:, 1].min():.1f} ~ {all_verts[:, 1].max():.1f} m")
```

---

### 10.6 提取自车未来轨迹

```python
import numpy as np

def extract_ego_fut_trajectory(scene_dir, frame_idx):
    """
    提取指定帧的自车未来轨迹（世界坐标系）。
    返回：
        poses: (M, 3) float32，从当前帧（含）到末帧的自车位置序列，
               每行 [x, y, z]，单位：米
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    entry = index['messages'][frame_idx]

    json_data, bin_data = parse_glb(scene_dir / entry['file'])
    nuviz  = json_data['nuviz']
    update = nuviz['data']['updates'][0]
    fut    = update['primitives'].get('/ego/fut_trajectory')
    if fut is None:
        return None

    traj = fut['trajectory'][0]
    poses = read_accessor(json_data, bin_data, traj['poses'])  # (M, 3)
    return poses

# 示例：读取第 5 帧自车未来轨迹并打印点数
poses = extract_ego_fut_trajectory('output/scene-0916', frame_idx=5)
if poses is not None:
    print(f'未来轨迹点数: {len(poses)}')          # = 总帧数 - 5
    print(f'当前帧位置:   {poses[0]}')            # index 0 = 当前帧
    print(f'末帧位置:     {poses[-1]}')
```

---

### 10.7 提取对象未来轨迹

```python
import numpy as np

def extract_objects_fut_trajectories(scene_dir, frame_idx):
    """
    提取指定帧所有对象的未来轨迹（CSR 格式解码）。
    返回：
        list of ndarray，长度 = 当前帧对象数 M；
        第 i 项为 (Li, 3) float32，该对象从当前帧起的未来中心点序列，
        Li 可为 0（对象在当前帧后不再出现）。
    """
    scene_dir = Path(scene_dir)
    index = json.loads((scene_dir / 'message_index.json').read_text())
    entry = index['messages'][frame_idx]

    json_data, bin_data = parse_glb(scene_dir / entry['file'])
    nuviz  = json_data['nuviz']
    update = nuviz['data']['updates'][0]
    fut    = update['primitives'].get('/objects/fut_trajectories')
    if fut is None:
        return None

    traj_meta = fut['trajectories'][0]
    obj_count = traj_meta['obj_count']
    points  = read_accessor(json_data, bin_data, traj_meta['points'])   # (T, 3)
    offsets = read_accessor(json_data, bin_data, traj_meta['offsets'])  # (M+1,)

    result = []
    for i in range(obj_count):
        s, e = int(offsets[i]), int(offsets[i + 1])
        result.append(points[s:e])  # (Li, 3)
    return result

# 示例：读取第 0 帧，打印各对象的未来轨迹点数
trajectories = extract_objects_fut_trajectories('output/scene-0916', frame_idx=0)
if trajectories is not None:
    for i, traj in enumerate(trajectories):
        print(f'对象 {i}: 未来 {len(traj)} 个轨迹点')

# 示例：结合 /objects/bounds 的 TRACK_ID 对应关系
objs = extract_objects('output/scene-0916', frame_idx=0)
if objs is not None and trajectories is not None:
    for i, (tid, traj) in enumerate(zip(objs['track_ids'], trajectories)):
        print(f'TRACK_ID={tid}: 未来 {len(traj)} 帧出现')
```

---

## 11. GLB 二进制布局详解

以一个包含点云+标注+1个相机图像的帧文件为例，BIN chunk 中数据的典型顺序如下：

```
BIN chunk
├── [bufferView 0]  LiDAR 点云 XYZ          float32, (N,3), ~480 KB
├── [padding]       4 字节对齐填充
├── [bufferView 1]  LiDAR intensity         float32, (N,),  ~120 KB
├── [padding]
├── [bufferView 2]  目标框 CENTER           float32, (M,3)
├── [padding]
├── [bufferView 3]  目标框 SIZE             float32, (M,3)
├── [padding]
├── [bufferView 4]  目标框 ROTATION         float32, (M,4)
├── [padding]
├── [bufferView 5]  目标框 CLASS_ID         uint32,  (M,)
├── [padding]
├── [bufferView 6]  目标框 TRACK_ID         uint32,  (M,)
├── [padding]
├── [bufferView 7]  CAM_FRONT JPEG 数据     bytes,   ~80 KB
├── [bufferView 8]  CAM_FRONT_LEFT JPEG     bytes,   ~80 KB
├── ...
├── [bufferView 12] CAM_BACK_RIGHT JPEG          bytes,   ~80 KB
├── [padding]
├── [bufferView 13] /ego/fut_trajectory poses        float32 (M, 3)
├── [padding]
├── [bufferView 14] /objects/fut_trajectories points   float32 (T, 3)
├── [padding]
└── [bufferView 15] /objects/fut_trajectories offsets  uint32  (M+1,)
```

每个 bufferView 通过 `byteOffset` + `byteLength` 精确定位其在 BIN chunk 中的位置，读取时：

```python
raw_bytes = bin_data[bv['byteOffset'] : bv['byteOffset'] + bv['byteLength']]
```

### accessor 与 bufferView 的关系

```
accessor
  └─ bufferView  ←─ byteOffset, byteLength 定位 BIN chunk 中的字节范围
       └─ componentType  决定 dtype（float32/uint32/...）
       └─ type           决定 shape（SCALAR/VEC3/VEC4/...）
       └─ count          元素个数
```

---

## 12. 模块文件索引

| 文件 | 职责 | 核心类/函数 |
|---|---|---|
| `converter.py` | 顶层转换入口，遍历 split 中的所有场景，调度 MetadataBuilder 和 MessageBuilder；预计算全场景 ego 轨迹与对象轨迹供各帧复用 | `NuScenesConverter` |
| `metadata_builder.py` | 构建 `metadata.glb`，提取相机内外参、场景描述、地图范围、类别映射、矢量地图 | `MetadataBuilder` |
| `message_builder.py` | 构建每帧 `messages/XXXXXX.glb`，处理 ego pose、LiDAR、目标框、相机图像、自车未来轨迹、对象未来轨迹 | `MessageBuilder` |
| `glb_encoder.py` | 底层 GLB 二进制编码器，管理 bufferViews、accessors、images 和 BIN chunk | `GLBEncoder` |
| `coord_utils.py` | 坐标变换工具：点云从传感器坐标系变换到世界坐标系、四元数格式转换 | `transform_points_to_world`, `quat_to_wxyz` |
| `visualize_map.py` | 从 `metadata.glb` 读取矢量地图并渲染为 PNG，叠加自车轨迹 | `visualize_scene_map` |

### 数据流图

```
NuScenes SDK + NuScenesMap
    │
    ├──────────────────────────────────────────────────────
    │  MetadataBuilder.build()                （每场景执行一次）
    │    ├── _build_cameras()  → 相机内外参 JSON
    │    ├── _build_map()      → 矢量地图 accessor  ← 地图集成
    │    │     ├── 收集 ego_pose 轨迹点
    │    │     ├── Shapely buffer(75m) → 缓冲多边形
    │    │     ├── NuScenesMap.get_records_in_patch()  粗筛
    │    │     ├── buffer_poly.intersects()            精筛
    │    │     ├── buffer_poly.intersection()          裁剪
    │    │     └── GLBEncoder.add_accessor() ×2/图层  → BIN chunk
    │    └── GLBEncoder.encode() → metadata.glb
    │
    └──────────────────────────────────────────────────────
       NuScenesConverter.convert_scene()  （每场景执行一次）
         ├── _collect_ego_poses()       → ego_all [[x,y,z], ...]  长度 N
         └── _collect_obj_trajectories() → obj_all {track_id: [(frame_idx,[x,y,z]), ...]}

       MessageBuilder.build_message()  × N帧  （每帧执行一次）
         ├── 读取 ego_pose  →  poses./ego_pose
         ├── _build_lidar()
         │     ├── 读取 .pcd.bin 点云文件
         │     ├── transform_points_to_world()（sensor→ego→world）
         │     └── GLBEncoder.add_accessor()  →  primitives./lidar
         ├── _build_objects()
         │     ├── 读取 sample_annotation（已在世界坐标系）
         │     └── GLBEncoder.add_accessor() ×5  →  primitives./objects/bounds
         ├── _build_camera() × 6
         │     ├── 读取相机图像文件
         │     ├── JPEG 压缩（quality=85）
         │     └── GLBEncoder.add_image()  →  primitives./camera/<CHANNEL>
         ├── _build_ego_fut_trajectory()
         │     └── ego_all[K:]  →  GLBEncoder.add_accessor()  →  primitives./ego/fut_trajectory
         ├── _build_objects_fut_trajectories()
         │     ├── _get_obj_future_for_frame()  → 按 track_id 过滤未来点
         │     └── GLBEncoder.add_accessor() ×2  →  primitives./objects/fut_trajectories
         └── GLBEncoder.encode() → messages/XXXXXX.glb
         （地图不出现在帧消息中）

visualize_map.py                         （离线可视化，按需运行）
    ├── parse_glb(metadata.glb)           → 读取矢量地图各图层
    ├── parse_glb(messages/*.glb) × N帧  → 读取自车轨迹
    └── matplotlib 渲染                  → map_<scene_name>.png
```

---

 