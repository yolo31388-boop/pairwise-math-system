"""
pairwise-math-system - MathSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class MathSystem:
    def __init__(self):
        self.state = {}

    def vector_dot(self, *args, **kwargs):
        """BUG: 向量点积不做维度检查，不同维度的向量也能算"""
        return None

    def matrix_multiply(self, *args, **kwargs):
        """BUG: 矩阵乘法不做维度检查，2x3乘4x5也能算"""
        return None

    def check_intersection(self, *args, **kwargs):
        """BUG: 碰撞检测只做AABB，不做球体和胶囊体"""
        return None

    def quaternion_multiply(self, *args, **kwargs):
        """BUG: 四元数乘法不做归一化，旋转后长度变了"""
        return None

    def interpolate(self, *args, **kwargs):
        """BUG: 插值只做线性，不做球面和样条"""
        return None

