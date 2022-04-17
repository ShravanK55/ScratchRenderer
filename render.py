"""
Module implementing classes and methods related to rendering.
"""

from entity import Camera, Light, Object
from geometry import Transformation, TransformStack
import json
import numpy as np


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
