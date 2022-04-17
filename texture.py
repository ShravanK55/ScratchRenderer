"""
Module that implements texture management.
"""

import math
import numpy as np
from PIL import Image, ImageOps


class TextureManager:
    """
    Modules that handles texture loading and retrieval.
    """

    def __init__(self):
        """
        Method to initialize the texture manager.
        """
        self.texture_cache = {}

    def load_texture(self, texture_path):
        """
        Method to load a texture from a file. If the texture was already loaded previously, returns the cached texture.

        Args:
            texture_path(str): Path to the texture file.

        Returns:
            (Image): Loaded texture.

        """
        texture = self.texture_cache.get(texture_path)

        if not texture:
            texture = Image.open(texture_path)
            texture = ImageOps.mirror(texture)
            self.texture_cache[texture_path] = texture

        return texture


def get_texture_color(texture, uv):
    """
    Method to get the color of a texture at a given UV co-ordinate using bilinear filtering.

    Args:
        texture(Image): Texture to get the color from.
        uv(list): UV co-ordinate.

    Returns:
        (list): Color at the given UV co-ordinate.

    """
    MAX_RGB = 255
    texture_width, texture_height = texture.size
    x = min(uv[0] * (texture_width - 1), texture_width - 2)
    y = min(uv[1] * (texture_height - 1), texture_height - 2)
    x = max(x, 0)
    y = max(y, 0)

    if isinstance(x, np.ndarray):
        x.astype(int)
        x = x[0]

    if isinstance(y, np.ndarray):
        y.astype(int)
        y = y[0]

    f = x - np.trunc(x)
    g = y - np.trunc(y)

    p00_color = np.array(texture.getpixel((math.trunc(x), math.trunc(y))))
    p11_color = np.array(texture.getpixel((math.trunc(x) + 1, math.trunc(y) + 1)))
    p10_color = np.array(texture.getpixel((math.trunc(x) + 1, math.trunc(y))))
    p01_color = np.array(texture.getpixel((math.trunc(x) + 1, math.trunc(y))))

    p0010_color = f * p10_color + (1 - f) * p00_color
    p0111_color = f * p11_color + (1 - f) * p01_color
    color = g * p0111_color + (1 - g) * p0010_color
    return color / MAX_RGB
