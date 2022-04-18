"""
Module implementing classes, methods and utilities for screen space ambient occlusion.
"""

import numpy as np


def lerp(a, b, f):
    """
    Method to perform linear interpolation between two values.

    Args:
        a(float): First value to perform LERP for.
        b(float): Second value to perform LERP for.
        f(float): LERP factor.

    Returns:
        (float): Linearly interpolated value.

    """
    return a + f * (b - a)


class SSAOKernel:
    """
    Module that implements an SSAO sample kernel.
    """

    def __init__(self, kernel_size=32):
        """
        Method to initialize the SSAO kernel.

        Args:
            kernel_size(int): Size of the SSAO sample kernel. Defaults to 32.

        """
        self.size = kernel_size
        self.samples = []

        for i in range(kernel_size):
            sample = np.array([np.random.random_sample() * 2.0 - 1.0, np.random.random_sample() * 2.0 - 1.0,
                               np.random.random_sample()])
            sample = sample / np.sqrt(np.dot(sample, sample))
            sample = sample * np.random.random_sample()

            scale = float(i) / kernel_size
            scale = lerp(0.1, 1.0, scale * scale)
            sample = sample * scale
            self.samples.append(sample)


class SSAONoise:
    """
    Module that implements noise for SSAO sampling.
    """

    def __init__(self, tile_size=4):
        """
        Method to initialize the SSAO noise.

        Args:
            tile_size(int): Tile size of the SSAO noise. Defaults to 4.

        """
        self.tile_size = tile_size
        self.noise = [[0 for _ in range(tile_size)] for _ in range(tile_size)]
        for j in range(tile_size):
            for i in range(tile_size):
                self.noise[i][j] = np.array([np.random.random_sample() * 2.0 - 1.0,
                                             np.random.random_sample() * 2.0 - 1.0,
                                             0.0])

    def sample(self, x, y):
        """
        Method to sample the noise texture.

        Args:
            x(float): X location to sample the noise from.
            y(float): Y location to sample the noise from.

        Returns:
            (list): Noise at the given sample point.

        """
        return self.noise[int(x) % self.tile_size][int(y) % self.tile_size]
