"""
Module for implementing utility functions.
"""

import numpy as np


def triangle_area(pos0, pos1, pos2):
    """
    Method to get the area of a triangle.
    This is equivalent to half the cross product of the lines the points in the triangle.
    Area = (0.5f) * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)).

    Args:
        pos0(list): Position vector of vertex 0.
        pos1(list): Position vector of vertex 1.
        pos2(list): Position vector of vertex 2.

    Returns:
        (float): Area of the triangle.

    """
    return 0.5 * ((pos0[0] * (pos1[1] - pos2[1])) + (pos1[0] * (pos2[1] - pos0[1])) + (pos2[0] * (pos0[1] - pos1[1])))
