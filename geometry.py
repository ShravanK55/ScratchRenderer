"""
Module implementing classes and methods related to geometry.
"""

import json
import math
import numpy as np


def translation_matrix(translation):
    """
    Method that returns a translation matrix.

    Args:
        translation(array): Translation vector.

    Returns:
        (array): Translation matrix.

    """
    return [[1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]]


def rotation_matrix(rotation):
    """
    Method that returns a rotation matrix.

    Args:
        rotation(array): Rotation vector.

    Returns:
        (array): Rotation matrix.

    """
    rotation_matrix = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]

    if rotation[0] != 0:
        angle = rotation[0]
        rx_matrix = [[1, 0,               0,                0],
                     [0, math.cos(angle), -math.sin(angle), 0],
                     [0, math.sin(angle), math.cos(angle),  0],
                     [0, 0,               0,                1]]
        np.matmul(rx_matrix, rotation_matrix)

    if rotation[1] != 0:
        angle = rotation[1]
        ry_matrix = [[math.cos(angle),  0, math.sin(angle), 0],
                     [0,                1, 0,               0],
                     [-math.sin(angle), 0, math.cos(angle), 0],
                     [0,                0, 0,               1]]
        np.matmul(ry_matrix, rotation_matrix)

    if rotation[2] != 0:
        angle = rotation[2]
        rz_matrix = [[math.cos(angle), -math.sin(angle), 0, 0],
                     [math.sin(angle), math.cos(angle),  0, 0],
                     [0,               0,                1, 0],
                     [0,               0,                0, 1]]
        np.matmul(rz_matrix, rotation_matrix)

    return rotation_matrix


def scale_matrix(scale):
    """
    Method that returns a scale matrix.

    Args:
        scale(array): Scale vector.

    Returns:
        (array): Scale matrix.

    """
    return [[scale[0], 0,        0,         0],
            [0,        scale[1], 0,         0],
            [0,        0,        scale[2],  0],
            [0,        0,        0,         1]]


class Transformation:
    """
    Class defining a transformation.
    """

    def __init__(self, translation=None, rotation=None, scale=None):
        """
        Method to initialize the transformation.

        Args:
            translation(array): Translation component of the transformation. Defaults to None.
            rotation(array): Rotation component of the transformation. Defaults to None.
            scale(array): Scale component of the transformation. Defaults to None.

        """
        self.matrix = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]

        if scale:
            self.apply_scale(scale)

        if rotation:
            self.apply_rotation(rotation)

        if translation:
            self.apply_translation(translation)

    def apply_translation(self, t):
        """
        Method to apply additional translation to the transformation.

        Args:
            t(array): Translation to apply.

        """
        np.matmul(translation_matrix(t), self.matrix)

    def apply_rotation(self, r):
        """
        Method to apply additional rotation to the transformation.

        Args:
            r(array): Rotation to apply.

        """
        np.matmul(rotation_matrix(r), self.matrix)

    def apply_scale(self, s):
        """
        Method to apply additional scale to the transformation.

        Args:
            s(array): Scale to apply.

        """
        np.matmul(scale_matrix(s), self.matrix)

    def get_inverse(self):
        """
        Method to get the inverse of the transformation matrix.

        Returns:
            (array): Inverse of the transformation matrix.

        """
        u = [self.matrix[0][0], self.matrix[1][0], self.matrix[2][0]]
        v = [self.matrix[0][1], self.matrix[1][1], self.matrix[2][1]]
        n = [self.matrix[0][2], self.matrix[1][2], self.matrix[2][2]]

        scale_x = np.sqrt(np.dot(u, u))
        scale_y = np.sqrt(np.dot(v, v))
        scale_z = np.sqrt(np.dot(n, n))

        t = [self.matrix[0][3], self.matrix[1][3], self.matrix[2][3]]

        u = [ui / scale_x for ui in u]
        v = [vi / scale_y for vi in v]
        n = [ni / scale_z for ni in n]

        inverse_mat = [[1 / scale_x, 0,           0,           0],
                       [0,           1 / scale_y, 0,           0],
                       [0,           0,           1 / scale_z, 0],
                       [0,           0,           0,           1]]

        # TODO: Check if this is the proper multiplication order.
        inverse_rot_mat = [[u[0], u[1], u[2], 0],
                           [v[0], v[1], v[2], 0],
                           [n[0], n[1], n[2], 0],
                           [0,    0,    0,    1]]
        inverse_mat = np.matmul(inverse_mat, inverse_rot_mat)

        # TODO: Check if this is the proper multiplication order.
        inverse_trans_mat = translation_matrix([-t[0], -t[1], -t[2]])
        inverse_mat = np.matmul(inverse_mat, inverse_trans_mat)

        return inverse_mat

    def get_transpose(self):
        """
        Method to get the transpose of the transformation matrix.

        Returns:
            (array): Transpose of the transformation matrix.

        """
        return np.transpose(self.matrix)

    def get_inverse_transpose(self, include_translation=False):
        """
        Method to get the inverse-transpose of the transformation matrix.

        Args:
            include_translation(bool): Whether to include translation in the inverse transpose. Defaults to False.

        Returns:
            (array): Inverse-transpose of the transformation matrix.

        """
        inverse_mat = self.get_inverse()

        if not include_translation:
            inverse_mat[0][3] = 0
            inverse_mat[1][3] = 0
            inverse_mat[2][3] = 0

        return np.transpose(inverse_mat)


class Vertex:
    """
    Class definining a vertex.
    """

    def __init__(self, pos, normal=None, uv=None):
        """
        Method to initialize the vertex.

        Args:
            pos(array): Position vector of the vertex.
            normal(array): Normal vector of the vertex.
            uv(array): UV vector of the vertex.

        """
        self.pos = pos
        self.normal = normal
        self.uv = uv


class Triangle:
    """
    Class definining a triangle.
    """

    def __init__(self, v0, v1, v2):
        """
        Method to initialize the triangle.

        Args:
            v0(Vertex): Vertex 0.
            v1(Vertex): Vertex 1.
            v2(Vertex): Vertex 2.

        """
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2


class Object:
    """
    Class defining an object.
    """

    def __init__(self, transformation, geometry=None, geometry_path=None):
        """
        Method to initialize an object.
        """
        self.transformation = transformation
        self.geometry = geometry if geometry else []

        if (not geometry) and (geometry_path is not None):
            self.import_geometry_from_file(geometry_path)

    def import_geometry_from_file(self, file_path):
        """
        Method to initialize the geometry of the object from a file.

        Args:
            file_path(str): Path of the file to import the geometry from.

        """
        self.geometry = []
        with open(file_path) as json_file:
            triangle_data = json.load(json_file)

        for triangle in triangle_data.get('data'):
            v0 = Vertex(triangle.get('v0').get('v'), triangle.get('v0').get('n'), triangle.get('v0').get('t'))
            v1 = Vertex(triangle.get('v1').get('v'), triangle.get('v1').get('n'), triangle.get('v1').get('t'))
            v2 = Vertex(triangle.get('v2').get('v'), triangle.get('v2').get('n'), triangle.get('v2').get('t'))
            tri = Triangle(v0, v1, v2)
            self.geometry.append(tri)
