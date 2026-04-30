# SparseDrive模型输出数据结构文档

## 概述

SparseDrive是一个端到端的自动驾驶感知和规划模型，能够同时预测：
- **3D目标检测**: 检测周围车辆、行人等目标
- **轨迹预测**: 预测检测目标的未来运动轨迹（多模态）
- **地图感知**: 预测道路地图元素（车道线、人行横道等）
- **运动规划**: 生成自车的规划轨迹（多模态）

## 数据文件信息

- **文件路径**: `/home/zhangxueyou/PycharmProjects/nuscenes-data-converter/data/sparsedrive/sparsedrive_stage2_trainval_with_metric.pkl`
- **文件格式**: Python Pickle (*.pkl)
- **总样本数**: 5119
- **数据结构**: 列表，每个元素对应一个时间帧的预测结果

## 顶层数据结构

```python
数据类型: list
样本数量: 5119
```

每个样本是一个字典，包含以下顶层keys:
```python
['img_bbox', 'metric_results']
```

实际的预测数据存储在 `img_bbox` 字段中。

## 预测数据字段详解 (img_bbox)


### boxes_3d
**说明**: 3D检测框，包含位置、尺寸、朝向等信息，shape: (N, 9)，其中N为检测到的目标数量。
        每个检测框包含9个值: [x, y, z, w, l, h, yaw, vx, vy]
        - x, y: 目标在ego坐标系下的横向和纵向位置（米）
        - z: 目标高度（米）
        - w, l, h: 目标的宽度、长度、高度（米）
        - yaw: 目标的朝向角（弧度）
        - vx, vy: 目标的速度（米/秒）
- **类型**: Tensor
- **形状**: [300, 10]
- **数据类型**: torch.float32
- **数值范围**: [-54.8881, 53.1856]
- **平均值**: 0.9771

### scores_3d
**说明**: 3D检测框的置信度分数，shape: (N,)，值域[0, 1]
- **类型**: Tensor
- **形状**: [300]
- **数据类型**: torch.float32
- **数值范围**: [0.0054, 0.5781]
- **平均值**: 0.0246

### labels_3d
**说明**: 3D检测框的类别标签，shape: (N,)，对应CLASSES中的索引
        类别包括: car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier
- **类型**: Tensor
- **形状**: [300]
- **数据类型**: torch.int64
- **数值范围**: [0.0000, 9.0000]
- **平均值**: 3.9133

### cls_scores
- **类型**: Tensor
- **形状**: [300]
- **数据类型**: torch.float32
- **数值范围**: [0.0225, 0.7457]
- **平均值**: 0.0491

### instance_ids
**说明**: 目标实例ID，用于跨帧跟踪，shape: (N,)
- **类型**: Tensor
- **形状**: [300]
- **数据类型**: torch.int64


### vectors
**说明**: 地图矢量预测，列表形式，每个元素是一个地图元素的点序列
        shape: List[(M, 2)]，其中M为每个地图元素的点数
- **类型**: list
- **长度**: 100
- **元素类型**: ndarray
- **元素形状**: [20, 2]

### scores
**说明**: 地图元素的置信度分数
- **类型**: ndarray
- **形状**: [100]
- **数据类型**: float32


### labels
**说明**: 地图元素的类别标签，对应MAP_CLASSES中的索引
        类别包括: ped_crossing（人行横道）, divider（分隔线）, boundary（边界）
- **类型**: ndarray
- **形状**: [100]
- **数据类型**: int64


### trajs_3d
**说明**: 多模态轨迹预测，shape: (N, 6, 12, 2)
        - N: 目标数量
        - 6: 每个目标预测6个模态轨迹
        - 12: 每个轨迹预测未来12个时间步（通常每步0.5秒，共6秒）
        - 2: 每个点的2D坐标 (x, y)
- **类型**: Tensor
- **形状**: [300, 6, 12, 2]
- **数据类型**: torch.float32


### trajs_score
**说明**: 轨迹置信度分数，shape: (N, 6)，每个目标的6个模态轨迹各有一个分数
- **类型**: Tensor
- **形状**: [300, 6]
- **数据类型**: torch.float32


### anchor_queue
- **类型**: Tensor
- **形状**: [300, 1, 10]
- **数据类型**: torch.float32


### period
- **类型**: Tensor
- **形状**: [300]
- **数据类型**: torch.int64


### planning_score
**说明**: 规划轨迹的置信度分数，shape: (3, 6)
- **类型**: Tensor
- **形状**: [3, 6]
- **数据类型**: torch.float32


### planning
**说明**: 多模态规划轨迹，shape: (3, 6, 12, 2)
        - 3: 三种驾驶指令（左转、直行、右转）
        - 6: 每个指令下预测6个模态轨迹
        - 12: 每个轨迹预测未来12个时间步
        - 2: 每个点的2D坐标 (x, y)
- **类型**: Tensor
- **形状**: [3, 6, 6, 2]
- **数据类型**: torch.float32


### final_planning
**说明**: 最终选定的规划轨迹，shape: (12, 2)，表示未来12个时间步的规划路径
- **类型**: Tensor
- **形状**: [6, 2]
- **数据类型**: torch.float32


### ego_period
- **类型**: Tensor
- **形状**: [1]
- **数据类型**: torch.int64


### ego_anchor_queue
- **类型**: Tensor
- **形状**: [1, 1, 10]
- **数据类型**: torch.float32


### token
**说明**: 样本的唯一标识符（sample token）
- **类型**: str
- **值**: 30e55a3ec6184d8cb1944b39ba19d622

### gt_ego_fut_cmd
- **类型**: ndarray
- **形状**: [3]
- **数据类型**: float32


### gt_ego_fut_trajs
- **类型**: ndarray
- **形状**: [6, 2]
- **数据类型**: float32


### ego_status
- **类型**: ndarray
- **形状**: [10]
- **数据类型**: float32


### gt_ego_fut_masks
- **类型**: ndarray
- **形状**: [6]
- **数据类型**: float32


### fut_boxes
- **类型**: list
- **长度**: 6
- **元素类型**: ndarray
- **元素形状**: [11, 7]

## 数据加载示例

### Python代码

```python
import pickle
import torch

def load_sparsedrive_predictions(file_path):
    # 强制CPU加载，避免CUDA问题
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    
    original_restore = torch.serialization.default_restore_location
    torch.serialization.default_restore_location = lambda storage, location: storage
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    finally:
        torch.cuda.is_available = original_is_available
        torch.serialization.default_restore_location = original_restore

# 加载数据
predictions = load_sparsedrive_predictions('sparsedrive_stage2_trainval_with_metric.pkl')

# 访问第一个样本
sample = predictions[0]['img_bbox']
sample_token = sample['token']
boxes_3d = sample['boxes_3d']  # 3D检测框
planning = sample['planning']  # 规划轨迹
```

## 坐标系统说明

### Ego坐标系
模型输出的所有坐标都是相对于自车（ego vehicle）的局部坐标系：
- **X轴**: 指向车辆前方
- **Y轴**: 指向车辆左侧
- **Z轴**: 指向车辆上方
- **原点**: 位于自车中心

### 坐标转换
如需将预测结果转换到全局坐标系，需要使用样本对应的ego_pose信息：
```python
from nuscenes.prediction.helper import convert_local_coords_to_global

# 假设已获取ego_pose
ego_translation = ego_pose['translation']
ego_rotation = ego_pose['rotation']

# 转换局部坐标到全局坐标
global_coords = convert_local_coords_to_global(
    local_coords, ego_translation, ego_rotation
)
```

## 类别定义

### 目标检测类别 (CLASSES)
```python
CLASSES = (
    'car',                    # 0: 小汽车
    'truck',                  # 1: 卡车
    'trailer',                # 2: 拖车
    'bus',                    # 3: 公交车
    'construction_vehicle',   # 4: 工程车辆
    'bicycle',                # 5: 自行车
    'motorcycle',             # 6: 摩托车
    'pedestrian',             # 7: 行人
    'traffic_cone',           # 8: 交通锥
    'barrier',                # 9: 护栏
)
```

### 地图元素类别 (MAP_CLASSES)
```python
MAP_CLASSES = (
    'ped_crossing',  # 0: 人行横道
    'divider',       # 1: 车道分隔线
    'boundary',      # 2: 道路边界
)
```

## 数据处理流程

参考 `temp/services/sparsedrive_service.py` 中的实现：

1. **加载预测结果**: 使用`load_prediction_data()`方法
2. **按token索引**: 将预测结果组织为 `sample_token -> prediction` 的映射
3. **坐标转换**: 将ego坐标系转换为全局坐标系
4. **阈值过滤**: 过滤置信度低于0.3的预测结果
5. **多模态处理**: 根据驾驶指令选择对应的规划轨迹

## 注意事项

1. **设备兼容性**: pkl文件可能包含GPU tensor，加载时需要处理设备映射
2. **内存占用**: 完整数据集较大，建议按需加载
3. **坐标一致性**: 注意区分局部坐标系和全局坐标系
4. **置信度阈值**: 建议使用0.3作为过滤阈值
5. **轨迹时间步**: 每个时间步通常对应0.5秒


