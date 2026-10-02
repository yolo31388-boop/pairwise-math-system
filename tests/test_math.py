import pytest
from math.system import MathSystem


class TestMathSystem:

    def test_vector_dot(self):
        """向量点积不做维度检查，不同维度的向量也能算"""
        obj = MathSystem()
        result = obj.vector_dot()
        self.assertIsNotNone(result)

    def test_matrix_multiply(self):
        """矩阵乘法不做维度检查，2x3乘4x5也能算"""
        obj = MathSystem()
        result = obj.matrix_multiply()
        self.assertIsNotNone(result)

    def test_check_intersection(self):
        """碰撞检测只做AABB，不做球体和胶囊体"""
        obj = MathSystem()
        result = obj.check_intersection()
        self.assertIsNotNone(result)

    def test_quaternion_multiply(self):
        """四元数乘法不做归一化，旋转后长度变了"""
        obj = MathSystem()
        result = obj.quaternion_multiply()
        self.assertIsNotNone(result)

    def test_interpolate(self):
        """插值只做线性，不做球面和样条"""
        obj = MathSystem()
        result = obj.interpolate()
        self.assertIsNotNone(result)

