"""
配置管理模块（优化版本）

集中管理所有硬编码的配置值，便于统一修改和维护

新增性能相关配置项
"""


class Config:
    """全局配置类"""
    
    # ===== 性能优化配置 =====
    # 是否默认启用pipeline优化
    USE_PIPELINE_BY_DEFAULT = False
    
    # 并行批量转换的默认线程数
    DEFAULT_MAX_WORKERS = 4
    
    # 地图元素插值点数（用于距离计算）
    MAP_INTERPOLATION_POINTS = 200
    
    # ===== 检测相关配置 =====
    # 检测置信度阈值
    DETECTION_SCORE_THRESHOLD = 0.3
    
    # 对象匹配距离阈值（米）
    OBJECT_MATCH_DISTANCE_THRESHOLD = 2.0
    
    # ===== 地图相关配置 =====
    # 默认patch大小（米）
    DEFAULT_PATCH_SIZE = [70, 100]
    
    # 地图边界扩展（米）
    MAP_MARGIN = 6.0
    
    # 地图提取时的扩展范围（米）
    MAP_EXTRACTION_BUFFER = 5.0
    
    # 地图边界收缩（米），用于提取边界时
    MAP_BOUNDARY_SHRINK = 0.1
    
    # ===== 轨迹预测相关配置 =====
    # 未来轨迹步数
    FUTURE_TRAJECTORY_STEPS = 12
    
    # 轨迹模态数量
    TRAJECTORY_MODES = 6
    
    # ===== 评估相关配置 =====
    # IoU阈值
    IOU_THRESHOLD = 0.5
    
    # 检测类别列表
    DETECTION_CLASSES = [
        "car", "truck", "trailer", "bus", "construction_vehicle",
        "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"
    ]
    
    # 地图元素类别
    MAP_CLASSES = ('ped_crossing', 'divider', 'boundary')
    
    # 地图匹配距离阈值（米）
    MAP_MATCH_THRESHOLD = 9.0
    
    # ===== 日志相关配置 =====
    # 是否显示详细进度
    VERBOSE = True
    
    # 是否显示调试信息
    DEBUG = False
    
    # ===== 缓存相关配置 =====
    # 是否启用pipeline缓存（用于性能优化）
    ENABLE_PIPELINE_CACHE = True
    
    # 单场景最大缓存帧数（0表示不限制）
    MAX_CACHED_FRAMES = 0
    
    # ===== 前端数据生成配置 =====
    # 是否生成场景索引文件
    GENERATE_SCENE_INDEX = False
    
    # 是否生成相机流数据
    GENERATE_CAMERA_STREAM = False
    
    # 缩略图数据简化采样率（每N帧采样一次）
    THUMBNAIL_SAMPLE_RATE = 3
    
    # 地图简化容差（Douglas-Peucker算法，米）
    MAP_SIMPLIFICATION_TOLERANCE = 2.0
    
    # ===== 相机图片相关配置 =====
    # 是否复制相机图片到场景目录
    COPY_CAMERA_IMAGES = False
    
    # 是否压缩相机图片
    COMPRESS_CAMERA_IMAGES = True
    
    # JPEG压缩质量 (1-100)
    CAMERA_IMAGE_QUALITY = 85
    
    # ===== 栅格地图相关配置 =====
    # 是否生成栅格底图
    GENERATE_BASEMAP = True
    
    # 底图渲染分辨率（像素/米）
    # 值越大，图片越清晰但文件越大
    BASEMAP_PIXELS_PER_METER = 10.0
    
    # 底图图层（用于渲染）
    BASEMAP_LAYERS = [
        'drivable_area',
        'road_segment', 
        'lane',
        'ped_crossing'
    ]
    
    # 底图输出格式
    BASEMAP_FORMAT = 'PNG'
    BASEMAP_DPI = 100


# 创建全局配置实例
config = Config()


