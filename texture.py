"""
Module that implements texture management.
"""

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
