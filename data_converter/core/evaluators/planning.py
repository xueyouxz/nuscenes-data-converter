"""
规划评估器

基于temp/services/evaluates/planning_evaluator.py重构
实现规划轨迹的L2误差和碰撞检测
"""

import numpy as np
from typing import Dict, List, Any
from shapely.geometry import Polygon


# 自车参数
EGO_WIDTH = 1.85
EGO_LENGTH = 4.084


class PlanningEvaluator:
    """
    规划评估器
    
    计算规划指标：L2误差、碰撞检测
    """
    
    def evaluate_sample(
        self,
        pred_planning: List[List[float]],
        gt_planning: List[List[float]],
        static_map: Dict[str, List] = None,
        gt_boxes: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        评估单帧规划结果
        
        Args:
            pred_planning: 预测规划轨迹
            gt_planning: GT规划轨迹
            static_map: 静态地图（用于碰撞检测）
            gt_boxes: GT框（用于碰撞检测）
            
        Returns:
            评估指标字典
        """
        # 计算L2误差
        l2_errors = self._compute_l2_errors(pred_planning, gt_planning)
        
        # 碰撞检测
        collision = False
        if static_map and gt_boxes:
            collision = self._detect_collision(pred_planning, static_map, gt_boxes)
        
        # 优化：只保留统计值，不存储完整的l2_errors列表（减少存储）
        mean_error = float(np.mean(l2_errors)) if l2_errors else 0.0
        max_error = float(np.max(l2_errors)) if l2_errors else 0.0
        min_error = float(np.min(l2_errors)) if l2_errors else 0.0
        
        return {
            "mean_l2_error": mean_error,
            "max_l2_error": max_error,
            "min_l2_error": min_error,
            "collision_detected": collision
        }
    
    def _compute_l2_errors(
        self,
        pred_traj: List[List[float]],
        gt_traj: List[List[float]]
    ) -> List[float]:
        """
        计算L2误差
        
        Args:
            pred_traj: 预测轨迹
            gt_traj: GT轨迹
            
        Returns:
            每个点的L2误差列表
            
        原理：
            计算每个时间步预测点与GT点之间的欧氏距离
        """
        if not pred_traj or not gt_traj:
            return []
        
        pred = np.array(pred_traj)
        gt = np.array(gt_traj)
        
        # 确保长度一致
        min_length = min(len(pred), len(gt))
        pred_trimmed = pred[:min_length]
        gt_trimmed = gt[:min_length]
        
        # 计算L2误差
        l2_errors = np.sqrt(((pred_trimmed - gt_trimmed) ** 2).sum(axis=1))
        
        return l2_errors.tolist()
    
    def _detect_collision(
        self,
        trajectory: List[List[float]],
        static_map: Dict[str, List],
        gt_boxes: List[Dict[str, Any]]
    ) -> bool:
        """
        检测碰撞
        
        Args:
            trajectory: 轨迹点列表
            static_map: 静态地图
            gt_boxes: GT框列表
            
        Returns:
            是否发生碰撞
            
        原理：
            1. 为轨迹上每个点创建自车多边形
            2. 检查是否与地图边界相交
            3. 检查是否与其他对象相交
        """
        if not trajectory or len(trajectory) < 2:
            return False
        
        # 排除起始点
        trajectory_np = np.array(trajectory[1:])
        
        # 创建轨迹多边形
        ego_polygons = self._create_trajectory_polygons(trajectory_np)
        
        # 检测地图碰撞
        if self._detect_map_collision(ego_polygons, static_map):
            return True
        
        # 检测对象碰撞
        if self._detect_object_collision(ego_polygons, gt_boxes):
            return True
        
        return False
    
    def _create_trajectory_polygons(
        self,
        trajectory: np.ndarray
    ) -> List[Polygon]:
        """
        为轨迹创建多边形序列
        
        Args:
            trajectory: 轨迹点 (T, 2)
            
        Returns:
            多边形列表
            
        原理：
            为每个轨迹点创建一个矩形多边形表示自车
            根据轨迹方向确定矩形朝向
        """
        polygons = []
        yaws = []
        trajectory_len = len(trajectory)
        
        if trajectory_len < 2:
            if trajectory_len == 1:
                return [self._create_ego_polygon(trajectory[0], 0.0)]
            return []
        
        # 计算每个点的朝向
        for i in range(trajectory_len):
            if i < trajectory_len - 1:
                # 使用当前点到下一点的方向
                next_point = trajectory[i + 1]
                dx = next_point[0] - trajectory[i][0]
                dy = next_point[1] - trajectory[i][1]
                yaw = np.arctan2(dy, dx)
            else:
                # 最后一个点使用前一个点的方向
                yaw = yaws[-1]
            yaws.append(yaw)
        
        # 创建多边形
        for point, yaw in zip(trajectory, yaws):
            poly = self._create_ego_polygon(point, yaw)
            polygons.append(poly)
        
        return polygons
    
    def _create_ego_polygon(
        self,
        center: np.ndarray,
        yaw: float
    ) -> Polygon:
        """
        创建自车多边形
        
        Args:
            center: 中心点 [x, y]
            yaw: 朝向角（弧度）
            
        Returns:
            多边形
        """
        half_length = EGO_LENGTH / 2
        half_width = EGO_WIDTH / 2
        
        # 四个角点（局部坐标系）
        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        # 旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # 旋转 + 平移
        rotated_corners = corners @ rotation_matrix.T
        world_corners = rotated_corners + center
        
        return Polygon(world_corners)
    
    def _detect_map_collision(
        self,
        ego_polygons: List[Polygon],
        static_map: Dict[str, List]
    ) -> bool:
        """
        检测地图碰撞
        
        Args:
            ego_polygons: 自车多边形列表
            static_map: 静态地图
            
        Returns:
            是否碰撞
        """
        # 提取边界
        boundaries = static_map.get('boundaries', [])
        if not boundaries:
            boundaries = static_map.get('boundary', [])
        
        # 检查每个自车多边形与边界的碰撞
        for ego_poly in ego_polygons:
            for boundary in boundaries:
                if self._check_polygon_boundary_collision(ego_poly, boundary):
                    return True
        
        return False
    
    def _detect_object_collision(
        self,
        ego_polygons: List[Polygon],
        gt_boxes: List[Dict[str, Any]]
    ) -> bool:
        """
        检测对象碰撞
        
        Args:
            ego_polygons: 自车多边形列表
            gt_boxes: GT框列表
            
        Returns:
            是否碰撞
        """
        for ego_poly in ego_polygons:
            for obj in gt_boxes:
                if self._check_polygon_object_collision(ego_poly, obj):
                    return True
        
        return False
    
    def _check_polygon_boundary_collision(
        self,
        ego_poly: Polygon,
        boundary_coords: List[List[float]]
    ) -> bool:
        """
        检查多边形与边界的碰撞
        
        Args:
            ego_poly: 自车多边形
            boundary_coords: 边界坐标点
            
        Returns:
            是否碰撞
        """
        from shapely.geometry import LineString
        
        if not boundary_coords or len(boundary_coords) < 2:
            return False
        
        boundary_line = LineString(boundary_coords)
        boundary_buffer = boundary_line.buffer(0.1)
        
        return ego_poly.intersects(boundary_buffer)
    
    def _check_polygon_object_collision(
        self,
        ego_poly: Polygon,
        obj_box: Dict[str, Any]
    ) -> bool:
        """
        检查多边形与对象的碰撞
        
        Args:
            ego_poly: 自车多边形
            obj_box: 对象框
            
        Returns:
            是否碰撞
        """
        obj_center = np.array(obj_box['translation'][:2])
        obj_yaw = obj_box.get('yaw', 0)
        obj_size = obj_box.get('size', [2, 2, 2])
        
        # 创建对象多边形
        obj_poly = self._create_object_polygon(obj_center, obj_yaw, obj_size[:2])
        
        return ego_poly.intersects(obj_poly)
    
    def _create_object_polygon(
        self,
        center: np.ndarray,
        yaw: float,
        size: List[float]
    ) -> Polygon:
        """
        创建对象多边形
        
        Args:
            center: 中心点
            yaw: 朝向
            size: 尺寸 [width, length]
            
        Returns:
            多边形
        """
        half_length = size[0] / 2
        half_width = size[1] / 2
        
        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        world_corners = rotated_corners + center
        
        return Polygon(world_corners)

