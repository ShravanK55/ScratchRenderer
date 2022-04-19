"""
Module for implementing utility functions.
"""

from constants import MAX_RGB
import numpy as np
import os


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


def smooth_step(l, r, x):
    """
    Method to perform a smooth clamping over a 0-1 interval using Hermite interpolation.

    Args:
        l(float): Lower bound for clamping.
        r(float): Upper bound for clamping.
        x(float): Parameter for smooth step.

    Returns:
        (float): Clamped smooth step value.

    """
    if (x < l):
        return 0

    if (x >= r):
        return 1

    x = (x - l) / (r - l)
    return x * x * (3 - 2 * x)


def save_image_to_ppm(image, out_file_name):
    """
    Method to save an image to a PPM file.

    Args:
        image(Image): Image to save.
        out_file_name(str): Output file name.

    """
    width, height = image.size

    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    ppm_file = open(out_file_name, "a")
    ppm_file.write("P3\n")
    ppm_file.write(str(width) + " " + str(height) + "\n")
    ppm_file.write(str(MAX_RGB) + "\n")
    pixel_string = ""
    for y in range(width):
        for x in range(height):
            coord = x, y
            pixel = list(image.getpixel(coord))
            if x == 0:
                pixel_string = pixel_string + str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2])
            else:
                pixel_string = pixel_string + " " + str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2])

        pixel_string = pixel_string + "\n"

    ppm_file.write(pixel_string)
    ppm_file.close()
