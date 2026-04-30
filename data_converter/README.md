# 数据转换器 (Data Converter)

将NuScenes数据集和SparseDrive模型预测转换为前端可视化所需的格式。

## 功能特性

### 生成的文件

为每个场景生成6个数据文件：

1. **metadata.json** - 场景元数据
   - 场景信息（名称、描述、帧数）
   - 自车状态（位姿、速度、加速度等）
   - 对象统计（每帧数量、类别分布）
   - 道路统计（交叉口、人行横道等）

2. **static_map.bin** - 静态地图（MessagePack格式）
   - 车道分隔线
   - 道路边界
   - 人行横道

3. **gt_stream.bin** - GT流数据（MessagePack格式）
   - 所有帧的GT对象
   - 对象位置、尺寸、速度等

4. **prediction_stream.bin** - 预测流数据（MessagePack格式）
   - 检测框
   - 规划轨迹
   - 在线地图预测

5. **metrics.json** - 评估指标
   - 检测指标（mAP、NDS等）
   - 地图指标（mAP、Chamfer距离）
   - 运动指标（ADE、FDE）
   - 规划指标（L2误差、碰撞率）

6. **associations.json** - 关联索引
   - GT到预测的映射
   - 预测到GT的映射

## 安装依赖

```bash
pip install msgpack numpy scipy shapely nuscenes-devkit torch
```

## 配置管理

所有硬编码的配置值已集中到 `config.py` 中，便于统一管理和修改：

```python
from data_converter import config

# 查看当前配置
print(config.DETECTION_SCORE_THRESHOLD)  # 0.3
print(config.OBJECT_MATCH_DISTANCE_THRESHOLD)  # 2.0

# 修改配置
config.DETECTION_SCORE_THRESHOLD = 0.5
config.VERBOSE = True
```

主要配置项：
- `DETECTION_SCORE_THRESHOLD`: 检测置信度阈值
- `OBJECT_MATCH_DISTANCE_THRESHOLD`: 对象匹配距离阈值（米）
- `MAP_MARGIN`: 地图边界扩展（米）
- `FUTURE_TRAJECTORY_STEPS`: 未来轨迹步数
- `VERBOSE`: 是否显示详细进度

## 使用方法

### 方法1：函数式接口（推荐）

```python
from data_converter import convert_scene

# 转换单个场景
result = convert_scene(
    scene_token='your_scene_token',
    nuscenes_dataroot='/path/to/nuscenes',
    sparsedrive_prediction='/path/to/predictions.pkl',
    output_dir='./output'
)
```

### 方法2：批量转换

```python
from data_converter import convert_scenes_batch

# 批量转换多个场景
results = convert_scenes_batch(
    scene_tokens=['token1', 'token2'],
    nuscenes_dataroot='/path/to/nuscenes',
    sparsedrive_prediction='/path/to/predictions.pkl',
    output_dir='./output'
)
```

### 方法3：使用类（更灵活）

```python
from data_converter import DataConverter

# 初始化转换器（只需一次）
converter = DataConverter(
    nuscenes_dataroot='/path/to/nuscenes',
    sparsedrive_prediction='/path/to/predictions.pkl',
    nuscenes_version='v1.0-trainval'
)

# 转换场景
result = converter.convert_scene(
    scene_token='your_scene_token',
    output_dir='./output',
    skip_metrics=False,  # 可选：跳过指标计算以加速
    skip_associations=False  # 可选：跳过关联计算
)
```

## 输出目录结构

```
output/
└── scene-0061/
    ├── metadata.json
    ├── static_map.bin
    ├── gt_stream.bin
    ├── prediction_stream.bin
    ├── metrics.json
    └── associations.json
```

## 模块架构

```
data_converter/
├── __init__.py              # 导出接口
├── converter.py             # 主协调器
├── utils/                   # 工具函数
│   ├── serialization.py     # MessagePack序列化
│   ├── coord_transform.py   # 坐标转换
│   └── statistics.py        # 统计计算
├── core/                    # 核心提取器
│   ├── nuscenes_extractor.py
│   ├── sparsedrive_extractor.py
│   ├── map_extractor.py
│   └── evaluators/          # 评估器
│       ├── detection.py
│       ├── mapping.py
│       ├── motion.py
│       └── planning.py
└── builders/                # 数据构建器
    ├── metadata_builder.py
    ├── map_builder.py
    ├── gt_stream_builder.py
    ├── prediction_stream_builder.py
    ├── metrics_builder.py
    └── associations_builder.py
```

## 性能优化

- 所有坐标数据统一使用全局坐标系，避免重复转换
- 使用MessagePack二进制格式，比JSON更高效
- 批量转换时复用提取器实例，减少初始化开销
- 可选择跳过耗时的指标计算

## 注意事项

1. 确保NuScenes数据集和SparseDrive预测文件路径正确
2. 指标计算较耗时，可根据需要跳过
3. 所有数据已转换到全局坐标系，前端可直接使用
4. MessagePack文件需要用相应的库读取

## 示例代码

参见项目根目录的 `example_usage.py` 文件。

