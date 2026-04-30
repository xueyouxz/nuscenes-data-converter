# NuScenes可视化数据结构规范

## 一、文件组织结构

```
output/
├── scenes_index.json              # 场景索引文件
├── types/                         # TypeScript类型定义
│   └── index.d.ts
└── scenes/                        # 各场景数据目录
    └── scene-XXXX/
        ├── metadata.json          # 场景元数据
        ├── static_map.bin         # 静态地图（MessagePack）
        ├── gt_stream.bin          # GT流数据（MessagePack）
        ├── prediction_stream.bin  # 预测流数据（MessagePack）
        ├── camera_stream.json     # 相机数据
        ├── metrics.json           # 评估指标
        ├── associations.json      # 关联索引
        ├── mapping_color.json     # 地图着色
        └── images/                # 相机图片
            ├── CAM_FRONT/
            │   ├── frame_000.jpg
            │   ├── frame_001.jpg
            │   └── ...
            ├── CAM_FRONT_LEFT/
            │   └── ...
            ├── CAM_FRONT_RIGHT/
            │   └── ...
            ├── CAM_BACK/
            │   └── ...
            ├── CAM_BACK_LEFT/
            │   └── ...
            └── CAM_BACK_RIGHT/
                └── ...
```

---

## 二、数据文件格式

### 1. scenes_index.json

**用途**：所有场景的索引文件，用于快速加载场景列表

**格式**：JSON

**数据结构**：
```typescript
{
  total_scenes: number;
  scenes: Array<{
    scene_token: string;
    scene_name: string;
    scene_description: string;
    frame_count: number;
    summary_metrics: {
      detection: DetectionMetrics;
      mapping: MappingMetrics;
      motion: MotionMetrics;
      planning: PlanningMetrics;
    };
  }>;
}
```

---

### 2. metadata.json

**用途**：场景元数据和所有帧的ego状态

**格式**：JSON

**数据结构**：
```typescript
{
  scene_token: string;
  scene_name: string;
  scene_description: string;
  frame_count: number;
  ego_states: Array<{
    frame_index: number;
    timestamp: number;
    pose: {
      translation: [number, number, number];
      rotation: [number, number, number, number];
    };
    state: {
      velocity: number;
      acceleration: number;
      steering_angle: number;
      yaw_rate: number;
    };
  }>;
  object_statistics: {
    total_unique_objects: number;
    objects_per_frame: number[];
    category_counts_per_frame: Record<string, number[]>;
    total_by_category: Record<string, number>;
  };
  road_statistics: {
    crosswalk_count: number;
    intersection_count: number;
    lane_count: number;
    traffic_light_count: number;
    stop_sign_count: number;
  };
}
```

**字段说明**：
- `object_statistics.total_unique_objects`：场景中唯一对象总数（基于instance_id去重）
- `object_statistics.objects_per_frame`：每帧的对象数量数组，长度等于frame_count
- `object_statistics.category_counts_per_frame`：每帧各类别对象数量，格式为 `{category: [count_per_frame]}`
- `object_statistics.total_by_category`：各类别唯一对象总数（基于instance_id去重），格式为 `{category: count}`
- `road_statistics.lane_count`：车道数量估算（基于divider数量+1）
- `road_statistics.intersection_count`：交叉口数量估算（基于车道和人行横道数量）
- `road_statistics.traffic_light_count`：红绿灯数量（从标注中提取）
- `road_statistics.stop_sign_count`：停止标志数量（从标注中提取）

---

### 3. static_map.bin

**用途**：场景静态地图元素（GT）

**格式**：MessagePack（二进制）

**数据结构**：
```typescript
{
  scene_token: string;
  divider: [number, number][][];
  boundary: [number, number][][];
  ped_crossing: [number, number][][];
  drivable_area: [number, number][][];
}
```

**说明**：
- 所有坐标使用全局坐标系
- `divider`：车道分隔线（lane_divider和road_divider），每个元素是一条折线的点序列
- `boundary`：道路边界线（从road_segment和lane的外轮廓提取），每个元素是一条折线的点序列
- `ped_crossing`：人行横道（从ped_crossing多边形的外轮廓提取），每个元素是一条折线的点序列
- `drivable_area`：可驾驶区域（从road_segment和lane合并后的多边形外轮廓提取），每个元素是一个多边形的点序列

---

### 4. gt_stream.bin

**用途**：所有帧的GT对象数据

**格式**：MessagePack（二进制）

**数据结构**：
```typescript
{
  scene_token: string;
  frames: Array<{
    frame_index: number;
    timestamp: number;
    ego_pose: {
      translation: [number, number, number];
      rotation: [number, number, number, number];
    };
    objects: {
      instance_ids: number[];
      categories: string[];
      boxes: Array<[number, number, number, number, number, number, number]>;
      velocities: [number, number][];
      distances_to_ego: number[];
      relative_velocities: [number, number][];
      visibility_levels: number[];
      visibility_descriptions: string[];
    };
  }>;
}
```

**说明**：
- 所有坐标使用全局坐标系
- box格式：[x, y, z, width, length, height, yaw]
- `instance_ids`：对象实例ID，类型为整数
- `distances_to_ego`：每个对象与自车的2D欧几里得距离（米）
- `relative_velocities`：每个对象相对于自车的速度（全局坐标系，米/秒）
- `visibility_levels`：可见性等级（0-4），0表示未知，1表示完全可见（80-100%），4表示大部分被遮挡（0-40%）
- `visibility_descriptions`：可见性描述，如'v80-100'（80-100%可见）、'v40-60'（40-60%可见）、'v0-40'（0-40%可见）等

---

### 5. prediction_stream.bin

**用途**：所有帧的模型预测数据

**格式**：MessagePack（二进制）

**数据结构**：
```typescript
{
  scene_token: string;
  frames: Array<{
    frame_index: number;
    timestamp: number;
    detection: {
      boxes: Array<[number, number, number, number, number, number, number]>;
      scores: number[];
      classes: string[];
      track_ids: string[];
      trajectories: Array<Array<[number, number][]>>;
    };
    planning: {
      trajectory: [number, number][];
    };
    mapping: {
      dividers: [number, number][][];
      boundaries: [number, number][][];
      ped_crossings: [number, number][][];
    };
  }>;
}
```

**说明**：
- 所有坐标使用全局坐标系
- box格式：[x, y, z, width, length, height, yaw]，其中z坐标固定为0.0
- `trajectories`：每个检测对象的轨迹预测，包含6个模态，每个模态12个时间步
- `planning.trajectory`：自车规划轨迹，包含未来若干个时间步的位置点（全局坐标）
- `mapping`：在线地图预测结果，包含dividers（车道分隔线）、boundaries（道路边界）、ped_crossings（人行横道）

---

### 6. camera_stream.json

**用途**：相机数据和图片路径

**格式**：JSON

**数据结构**：
```typescript
{
  scene_token: string;
  frames: Array<{
    frame_index: number;
    timestamp: number;
    cameras: Array<{
      channel: string;
      image_path: string;
      width: number;
      height: number;
      intrinsic: [[number, number, number], [number, number, number], [number, number, number]];
      extrinsic: {
        translation: [number, number, number];
        rotation: [number, number, number, number];
      };
      ego_to_camera: number[][];
    }>;
  }>;
}
```

**说明**：
- 相机图片按相机通道组织在`images/{channel}/`目录
- 图片文件名格式：`frame_{frame_index:03d}.jpg`
- 相机通道：CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
- `intrinsic`：相机内参矩阵（3x3）
- `extrinsic`：相机外参（相机在ego坐标系中的位姿）
- `ego_to_camera`：预计算的从ego坐标系到camera坐标系的4x4变换矩阵

---

### 7. metrics.json

**用途**：场景和帧级别的评估指标

**格式**：JSON

**数据结构**：
```typescript
{
  scene_token: string;
  scene_summary: {
    detection: {
      NDS: number;
      mAP: number;
      mATE: number;
      mASE: number;
      mAOE: number;
      mAVE: number;
      mAAE: number;
    };
    mapping: {
      AP_ped: number;
      AP_divider: number;
      AP_boundary: number;
      mAP: number;
    };
    motion: {
      minADE: number;
      minFDE: number;
      MR: number;
      EPA: number;
    };
    planning: {
      mean_l2_error: number;
      max_l2_error: number;
      collision_rate: number;
    };
  };
  frame_metrics: Array<{
    frame_index: number;
    timestamp: number;
    detection: {
      NDS: number;
      mAP: number;
      mATE: number;
      mASE: number;
      mAOE: number;
      mAVE: number;
      mAAE: number;
      AP_by_class: Record<string, number>;
      num_tp: number;
      num_fp: number;
      num_fn: number;
    };
    mapping: {
      mAP: number;
      AP_by_class: Record<string, number>;
    };
    motion: {
      minADE: number;
      minFDE: number;
      MR: number;
      EPA: number;
    };
    planning: {
      mean_l2_error: number;
      max_l2_error: number;
      min_l2_error: number;
      collision_detected: boolean;
    };
  }>;
}
```

**字段说明**：
- `detection.AP_by_class`：各检测类别的AP值，类别包括：car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier
- `detection.num_tp`：真正例（True Positive）数量
- `detection.num_fp`：假正例（False Positive）数量
- `detection.num_fn`：假负例（False Negative）数量
- `mapping.AP_by_class`：各地图类别的AP值，包括：ped_crossing, divider, boundary
- `motion.minADE`：最小平均位移误差（Minimum Average Displacement Error）
- `motion.minFDE`：最小最终位移误差（Minimum Final Displacement Error）
- `motion.MR`：未命中率（Miss Rate）
- `motion.EPA`：端点精度（Endpoint Accuracy）
- `planning.min_l2_error`：最小L2误差（单个时间步的最小误差）
- `planning.collision_detected`：是否检测到碰撞

---

### 8. associations.json

**用途**：GT与预测的关联关系

**格式**：JSON

**数据结构**：
```typescript
{
  scene_token: string;
  object_associations: Array<{
    frame_index: number;
    pred_instance_id: number | string;
    gt_instance_id?: number | string;
    category: string;
    score: number;
    iou?: number;
    distance?: number;
    is_tp: boolean;
    errors?: {
      translation_error: number;
      scale_error: number;
      orientation_error: number;
      velocity_error: number;
      attribute_error: number;
    };
  }>;
  map_associations: Array<{
    frame_index: number;
    pred_index: number;
    gt_element_id?: string;
    category: string;
    score: number;
    chamfer_distance?: number;
    min_distance?: number;
  }>;
  indexes: {
    gt_to_pred_objects: Record<string, Record<number, number[]>>;
    pred_to_gt_objects: Record<string, Record<number, number>>;
    gt_to_pred_map_elements: Record<string, Record<number, number[]>>;
    pred_to_gt_map_elements: Record<string, string>;
  };
}
```

**说明**：
- `map_associations`中使用`pred_index`而不是`pred_element_id`，表示预测地图元素在当前帧中的索引
- `map_associations`中同时包含`chamfer_distance`和`min_distance`字段（值相同）
- `indexes.gt_to_pred_objects`和`indexes.gt_to_pred_map_elements`的值是嵌套的对象，第一层key是instance_id/element_id，第二层key是frame_index

#### 注:
数据结构中indexes属性的对象示例如下：

```json
{
  "gt_to_pred_objects": {
    "113": {
      "0": [519873],
      "1": [519873],
      "2": [519873]
    },
    "114": {
      "0": [519700],
      "1": [519700]
    }
  },
  "pred_to_gt_objects": {
    "519873": {
      "0": 113,
      "1": 113,
      "2": 113,
      "3": 113,
      "4": 113
    },
    "519700": {
      "0": 114,
      "1": 114,
      "2": 114,
      "3": 114,
      "4": 114,
      "5": 114
    }
  },
  "gt_to_pred_map_elements": {
    "divider:5": {
      "0": [2, 5],
      "1": [3]
    },
    "boundary:10": {
      "0": [1],
      "2": [0, 2]
    }
  },
  "pred_to_gt_map_elements": {
    "0:divider:2": "divider:5",
    "0:divider:5": "divider:5",
    "1:divider:3": "divider:5",
    "0:boundary:1": "boundary:10",
    "2:boundary:0": "boundary:10"
  }
}
```

**字段说明**：
- `gt_to_pred_objects`：从GT实例ID到预测实例ID的映射，格式为 `{gt_instance_id: {frame_index: [pred_instance_ids]}}`
- `pred_to_gt_objects`：从预测实例ID到GT实例ID的映射，格式为 `{pred_instance_id: {frame_index: gt_instance_id}}`
- `gt_to_pred_map_elements`：从GT地图元素ID到预测元素索引的映射，格式为 `{gt_element_id: {frame_index: [pred_indices]}}`
- `pred_to_gt_map_elements`：从预测元素标识到GT元素ID的映射，预测元素标识格式为 `{frame_index}:{category}:{pred_index}`
---

### 9. mapping_color.json

**用途**：GT地图元素的着色信息（基于预测误差）

**格式**：JSON

**数据结构**：
```typescript
{
  scene_token: string;
  colored_elements: {
    divider: Array<{
      element_id: string;
      coordinates: [number, number][];
      point_errors: number[];
      avg_error: number;
      max_error: number;
      min_error: number;
      prediction_count: number;
      coverage_ratio: number;
    }>;
    boundary: Array<{...}>;
    ped_crossing: Array<{...}>;
  };
}
```

**字段说明**：
- `element_id`：元素唯一标识，格式为`{category}:{index}`
- `coordinates`：GT地图元素的坐标点序列
- `point_errors`：每个坐标点的预测误差（米，基于Chamfer距离）
- `avg_error`：所有点的平均误差
- `max_error`：最大误差点
- `min_error`：最小误差点
- `prediction_count`：跨所有帧，匹配到该GT元素的预测总数
- `coverage_ratio`：被预测覆盖的点的比例（0-1）

---

## 三、数据关系说明

### 3.1 文件依赖关系

```
scenes_index.json
    ↓ (引用)
scene-XXXX/metadata.json
    ↓ (时间戳对应)
scene-XXXX/gt_stream.bin
scene-XXXX/prediction_stream.bin
scene-XXXX/camera_stream.json
    ↓ (关联)
scene-XXXX/associations.json
scene-XXXX/metrics.json
    ↓ (基于GT地图)
scene-XXXX/static_map.bin
scene-XXXX/mapping_color.json
```

### 3.2 坐标系统

**全局坐标系**：
- 所有空间数据（地图、对象位置）使用全局坐标系
- 单位：米
- 原点：场景特定

**Ego坐标系**：
- 以自车为原点的局部坐标系
- 需要通过ego_pose进行转换

**Camera坐标系**：
- 以相机为原点的局部坐标系
- 使用预计算的ego_to_camera矩阵进行转换

---


## 七、存储格式选择

### 7.1 JSON vs MessagePack

**使用JSON的文件**：
- scenes_index.json
- metadata.json
- camera_stream.json
- metrics.json
- associations.json
- mapping_color.json


**使用MessagePack的文件**：
- static_map.bin
- gt_stream.bin
- prediction_stream.bin



### 7.2 图片组织方式

**按相机通道组织**（采用方案）：
```
images/
├── CAM_FRONT/
│   ├── frame_000.jpg
│   └── frame_001.jpg
└── ...
```

---



