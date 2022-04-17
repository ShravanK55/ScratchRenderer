"""
Module implementing classes and methods related to rendering.
"""

from entity import Camera, Light, Object
from geometry import Transformation, TransformStack
import json
import numpy as np


class GeometryBuffer:
    """
    Module that implements a geometry buffer.
    """

    def __init__(self, resolution):
        """
        Method to initialize the geometry buffer.

        Args:
            resolution(list): Resolution of the geometry buffer.

        """
        self.position_buffer = [[[0, 0, 0] for _ in range(resolution[0])] for _ in range(resolution[1])]
        self.color_buffer = [[[0, 0, 0] for _ in range(resolution[0])] for _ in range(resolution[1])]
        self.normal_buffer = [[[0, 0, 0] for _ in range(resolution[0])] for _ in range(resolution[1])]
        self.depth_buffer = np.matrix(np.ones((resolution[0], resolution[1])) * np.inf)

    def get_attributes(self, x, y):
        """
        Method to get the attributes at a point in the geometry buffer.

        Args:
            x(int): X position in the geometry buffer.
            y(int): Y position in the geometry buffer.

        Returns:
            (tuple): Tuple containing a position vector, color vector, normal vector and a depth value from the geometry
                buffer.

        """
        return (self.get_position(x, y), self.get_color(x, y), self.get_normal(x, y), self.get_depth(x, y))

    def set_attributes(self, x, y, position=None, color=None, normal=None, depth=np.inf):
        """
        Method to set the attributes at a point in the geometry buffer.

        Args:
            x(int): X position in the geometry buffer.
            y(int): Y position in the geometry buffer.
            position(list): Position vector. Defaults to None.
            color(list): Color vector. Defaults to None.
            normal(list): Normal vector. Defaults to None.
            depth(float): Depth value. Defaults to np.inf.

        Returns:
            (bool): Whether the attributes were updated in the geometry buffer.

        """
        success = self.set_depth(x, y, depth)
        if success:
            if position:
                self.set_position(x, y, position)

            if color:
                self.set_color(x, y, color)

            if normal:
                self.set_normal(x, y, normal)

        return success

    def get_position(self, x, y):
        """
        Method to get a position on the position buffer.

        Args:
            x(int): X position in the position buffer.
            y(int): Y position in the position buffer.

        Returns:
            (list): Position vector from the position buffer.

        """
        return self.position_buffer[y][x]

    def set_position(self, x, y, position):
        """
        Method to set a position on the position buffer.

        Args:
            x(int): X position in the position buffer.
            y(int): Y position in the position buffer.
            position(list): Position vector.

        """
        self.position_buffer[y][x] = position

    def get_color(self, x, y):
        """
        Method to get a color on the color buffer.

        Args:
            x(int): X position in the color buffer.
            y(int): Y position in the color buffer.

        Returns:
            (list): Color from the color buffer.

        """
        return self.color_buffer[y][x]

    def set_color(self, x, y, color):
        """
        Method to set a color on the color buffer.

        Args:
            x(int): X position in the color buffer.
            y(int): Y position in the color buffer.
            color(list): Color vector.

        """
        self.color_buffer[y][x] = color

    def get_normal(self, x, y):
        """
        Method to get a normal on the normal buffer.

        Args:
            x(int): X position in the normal buffer.
            y(int): Y position in the normal buffer.

        Returns:
            (list): Normal vector from the normal buffer.

        """
        return self.normal_buffer[y][x]

    def set_normal(self, x, y, normal):
        """
        Method to set a normal on the normal buffer.

        Args:
            x(int): X position in the normal buffer.
            y(int): Y position in the normal buffer.
            normal(list): Normal vector.

        """
        self.normal_buffer[y][x] = normal

    def get_depth(self, x, y):
        """
        Method to get a depth value on the depth buffer.

        Args:
            x(int): X position in the depth buffer.
            y(int): Y position in the depth buffer.

        Returns:
            (float): Depth value from the depth buffer.

        """
        return self.depth_buffer[y][x]

    def set_depth(self, x, y, depth):
        """
        Method to set a depth value on the depth buffer.

        Args:
            x(int): X position in the depth buffer.
            y(int): Y position in the depth buffer.
            depth(float): Depth value.

        Returns:
            (bool): Whether the depth value was updated in the depth buffer.

        """
        if depth < self.depth_buffer[y][x]:
            self.depth_buffer[y][x] = depth
            return True

        return False


class Renderer:
    """
    Module that implements a renderer.
    """

    def __init__(self, scene_file_path=None):
        """
        Method to initialize the renderer.

        Args:
            scene_file_path(str): File path of the scene. Defaults to None.

        """
        self.scene = {}
        self.camera = None
        self.lights = []
        self.objects = []
        self.transform_stack = TransformStack()
        self.light_transform_stack = TransformStack()
        self.geometry_buffer = GeometryBuffer((1, 1))

        if scene_file_path:
            self.load_scene(scene_file_path)

    def load_scene(self, scene_file_path):
        """
        Method to load a scene to the renderer.

        Args:
            scene_file_path(str): File path of the scene. Defaults to None.

        """
        with open(scene_file_path) as json_file:
            self.scene = json.load(json_file).get("scene")

        # Getting camera data
        camera_data = self.scene.get("camera")
        camera_position = camera_data.get("from")
        camera_look_at = camera_data.get("to")
        camera_bounds = camera_data.get("bounds")
        camera_resolution = camera_data.get("resolution")
        camera_direction = [camera_position[i] - camera_look_at[i] for i in range(3)]
        camera_direction = camera_direction / np.sqrt(np.dot(camera_direction, camera_direction))

        self.camera = Camera(camera_position, camera_direction, camera_bounds, camera_resolution)
        self.geometry_buffer = GeometryBuffer(camera_resolution)

        self.transform_stack = TransformStack()
        self.transform_stack.push(self.camera.projection_matrix)
        self.transform_stack.push(self.camera.cam_matrix)

        # Getting lights data
        self.lights = []
        self.light_transform_stack = TransformStack()
        lights_data = self.scene.get("lights")
        for light_data in lights_data:
            light_type = light_data.get("type", "ambient")
            light_color = light_data.get("color", [0, 0, 0])
            light_position = light_data.get("from")
            light_look_at = light_data.get("to")
            light_bounds = light_data.get("bounds")
            light_resolution = light_data.get("resolution")

            light_direction = None
            if light_position and light_look_at:
                light_direction = [light_position[i] - light_look_at[i] for i in range(3)]
                light_direction = light_direction / np.sqrt(np.dot(light_direction, light_direction))

            light = Light(light_type, light_color, light_position, light_direction, light_bounds, light_resolution)
            self.lights.append(light)

        # Gettting the objects in the scene.
        self.objects = []
        objects_data = self.scene.get("shapes")
        for object_data in objects_data:
            geometry_path = object_data.get("geometry") + ".json"
            transform_stack = TransformStack()
            for transform in object_data.get("transforms", []):
                t = Transformation(transform.get("T"), [transform.get("Rx"), transform.get("Ry"), transform.get("Rz")],
                                   transform.get("S"))
                transform_stack.push(t)
            object_transform = transform_stack.top()

            material = object_data.get("material")
            color = material.get("Cs")
            ka = material.get("Ka")
            kd = material.get("Kd")
            ks = material.get("Ks")
            specularity = material.get("n")
            texture_path = material.get("texture")

            obj = Object(object_transform, geometry_path, color, ka, kd, ks, specularity, texture_path)
            self.objects.append(obj)
