"""
Module implementing objects that exist in a scene.
"""

from geometry import Triangle, Vertex
import numpy as np
import json


class Object:
    """
    Class defining an object.
    """

    def __init__(self, transformation, geometry_path=None, color=None, ka=0.0, kd=0.0, ks=0.0, kt=0.0,
                 specularity=0.0, texture=None, normal_map=None):
        """
        Method to initialize an object.

        Args:
            transformation(Transformation): Transformation of the object.
            geometry_path(str): Path of the file to import the geometry from. Defaults to None.
            color(list): Flat color of the object. Defaults to None.
            ka(float): Ambient co-efficient. Defaults to 0.0.
            kd(float): Diffuse co-efficient. Defauls to 0.0.
            ks(float): Specular co-efficient. Defaults to 0.0.
            kt(float): Texture co-efficient. Defaults to 0.0.
            specularity(float): Specularity of the object. Defaults to 0.0.
            texture(Image): Texture to use for the object. Defaults to None.
            normal_map(Image): Normal map to use for the object. Defaults to None.

        """
        self.transformation = transformation
        self.geometry = []
        self.color = color if color else [0, 0, 0]
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.kt = kt
        self.specularity = specularity
        self.texture = texture
        self.normal_map = normal_map

        if geometry_path:
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


class Camera:
    """
    Class defining a camera for the scene.
    """

    def __init__(self, position, direction, frustum_bounds, resolution):
        """
        Method to initialize a camera.

        Args:
            position(list): Position that the camera is at.
            direction(list): Direction that the camera is facing.
            frustum_bounds(list): Bounds of the camera frustum. Format: [Near, Far, Right, Left, Top, Bottom].
            resolution(list): Resolution of the camera.

        """
        self.position = position
        self.direction = direction
        self.near, self.far, self.right, self.left, self.top, self.bottom = frustum_bounds
        self.resolution = resolution

        self.cam_matrix = self.generate_cam_matrix()
        self.projection_matrix = self.generate_projection_matrix()

    def generate_cam_matrix(self):
        """
        Generates a camera matrix from the camera parameters.

        Returns:
            (matrix): Camera matrix for conversion to camera space.

        """
        n = self.direction
        world_up = [0, 1, 0]

        u = np.cross(world_up, n)
        u = u / np.sqrt(np.dot(u, u))

        v = np.cross(n, u)
        v = v / np.sqrt(np.dot(v, v))

        return [[u[0], u[1], u[2], -np.dot(u, self.position)],
                [v[0], v[1], v[2], -np.dot(v, self.position)],
                [n[0], n[1], n[2], -np.dot(n, self.position)],
                [0,    0,    0,    1]]

    def generate_projection_matrix(self):
        """
        Generates a perspective projection matrix from the camera parameters.

        Returns:
            (matrix): Perspective projection matrix for conversion to NDC space.

        """
        return [[2 * self.near / (self.right - self.left), 0, (self.right + self.left) / (self.right - self.left), 0],
                [0, 2 * self.near / (self.top - self.bottom), (self.top + self.bottom) / (self.top - self.bottom), 0],
                [0, 0, -((self.far + self.near) / (self.far - self.near)), -((2 * self.far * self.near) / (self.far - self.near))],
                [0, 0, -1, 0]]


class Light:
    """
    Class defining a ambient/directional light source.
    """

    def __init__(self, light_type, color, intensity=1.0, position=None, direction=None, frustum_bounds=None,
                 resolution=None):
        """
        Method to initialize a light source.

        Args:
            light_type(str): Type of light source. Can either be "ambient" or "directional".
            color(list): Color of the light.
            intensity(float): Intensity of the light. Defaults to 1.0.
            position(list): Position of the light source. Defaults to None.
            direction(list): Direction of the light source. Defaults to None.
            frustum_bounds(list): Frustum bounds of the light source. Used for shadow mapping. Defaults to None.
            resolution(list): Resolution of the light source. Used for shadow mapping. Defaults to None.

        """
        self.type = light_type
        self.color = color
        self.intensity = intensity
        self.position = position
        self.direction = direction
        self.view_space_direction = direction
        self.camera = None if self.type == "ambient" else Camera(position, direction, frustum_bounds, resolution)
        self.shadow_buffer = None

        # Max depth is used for visualization purposes only.
        self.max_depth = np.inf

        if self.type == "directional":
            self.shadow_buffer = np.matrix(np.ones((self.camera.resolution[0], self.camera.resolution[1])) * np.inf)

    def set_view_space_direction(self, view_matrix):
        """
        Method to set the view space direction for the light.

        Args:
            view_matrix(matrix): View matrix to transform the world space light direction.

        """
        direction = [[self.direction[0]], [self.direction[1]], [self.direction[2]], [0]]
        self.view_space_direction = np.matmul(view_matrix, direction)
        self.view_space_direction = np.transpose(self.view_space_direction[:-1])[0]
        self.view_space_direction = self.view_space_direction / np.sqrt(
            np.dot(self.view_space_direction, self.view_space_direction))

    def clear_shadow_buffer(self):
        """
        Method to clear the shadow buffer.
        """
        self.shadow_buffer = np.matrix(np.ones((self.camera.resolution[0], self.camera.resolution[1])) * np.inf)

        # Max depth is used for visualization purposes only.
        self.max_depth = np.NINF

    def get_shadow_buffer_depth(self, x, y):
        """
        Method to get a depth value on the shadow buffer.

        Args:
            x(int): X position in the shadow buffer.
            y(int): Y position in the shadow buffer.

        Returns:
            (float): Depth value from the shadow buffer.

        """
        return self.shadow_buffer[x, y]

    def set_shadow_buffer(self, x, y, depth):
        """
        Method to set a depth value on the shadow buffer.

        Args:
            x(int): X position in the shadow buffer.
            y(int): Y position in the shadow buffer.
            depth(float): Depth value.

        Returns:
            (bool): Whether the depth value was updated in the shadow buffer.

        """
        if depth < self.shadow_buffer[x, y]:
            self.shadow_buffer[x, y] = depth
            if depth > self.max_depth:
                self.max_depth = depth
            return True

        return False
