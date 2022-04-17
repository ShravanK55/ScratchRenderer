"""
Module implementing classes and methods related to rendering.
"""

from entity import Camera, Light, Object
from geometry import Transformation, TransformStack
import json
import math
import numpy as np
from PIL import Image
from shader import fragment_shader
from texture import TextureManager


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
        self.position_buffer = [[[0, 0, 0] for _ in range(resolution[1])] for _ in range(resolution[0])]
        self.color_buffer = [[[0, 0, 0] for _ in range(resolution[1])] for _ in range(resolution[0])]
        self.normal_buffer = [[[0, 0, 0] for _ in range(resolution[1])] for _ in range(resolution[0])]
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
        return self.position_buffer[x][y]

    def set_position(self, x, y, position):
        """
        Method to set a position on the position buffer.

        Args:
            x(int): X position in the position buffer.
            y(int): Y position in the position buffer.
            position(list): Position vector.

        """
        self.position_buffer[x][y] = position

    def get_color(self, x, y):
        """
        Method to get a color on the color buffer.

        Args:
            x(int): X position in the color buffer.
            y(int): Y position in the color buffer.

        Returns:
            (list): Color from the color buffer.

        """
        return self.color_buffer[x][y]

    def set_color(self, x, y, color):
        """
        Method to set a color on the color buffer.

        Args:
            x(int): X position in the color buffer.
            y(int): Y position in the color buffer.
            color(list): Color vector.

        """
        self.color_buffer[x][y] = color

    def get_normal(self, x, y):
        """
        Method to get a normal on the normal buffer.

        Args:
            x(int): X position in the normal buffer.
            y(int): Y position in the normal buffer.

        Returns:
            (list): Normal vector from the normal buffer.

        """
        return self.normal_buffer[x][y]

    def set_normal(self, x, y, normal):
        """
        Method to set a normal on the normal buffer.

        Args:
            x(int): X position in the normal buffer.
            y(int): Y position in the normal buffer.
            normal(list): Normal vector.

        """
        self.normal_buffer[x][y] = normal

    def get_depth(self, x, y):
        """
        Method to get a depth value on the depth buffer.

        Args:
            x(int): X position in the depth buffer.
            y(int): Y position in the depth buffer.

        Returns:
            (float): Depth value from the depth buffer.

        """
        return self.depth_buffer[x][y]

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
        if depth < self.depth_buffer[x][y]:
            self.depth_buffer[x][y] = depth
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
        self.geometry_buffer = GeometryBuffer((1, 1))
        self.texture_manager = TextureManager()

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
        lights_data = self.scene.get("lights")
        for light_data in lights_data:
            light_type = light_data.get("type", "ambient")
            light_color = light_data.get("color", np.array([0, 0, 0]))
            light_intensity = light_data.get("intensity", 1.0)
            light_position = light_data.get("from")
            light_look_at = light_data.get("to")
            light_bounds = light_data.get("bounds")
            light_resolution = light_data.get("resolution")

            light_direction = None
            if light_position and light_look_at:
                light_direction = [light_position[i] - light_look_at[i] for i in range(3)]
                light_direction = light_direction / np.sqrt(np.dot(light_direction, light_direction))

            light = Light(light_type, light_color, light_intensity, light_position, light_direction, light_bounds,
                          light_resolution)
            self.lights.append(light)

        # Gettting the data of the objects in the scene.
        self.objects = []
        objects_data = self.scene.get("shapes")
        for object_data in objects_data:
            geometry_path = object_data.get("geometry") + ".json"
            transformation = Transformation()
            for transform in object_data.get("transforms", []):
                if transform.get("S"):
                    transformation.apply_scale(transform.get("S"))
                elif transform.get("Rx"):
                    transformation.apply_rotation([transform.get("Rx"), 0, 0])
                elif transform.get("Ry"):
                    transformation.apply_rotation([0, transform.get("Ry"), 0])
                elif transform.get("Rz"):
                    transformation.apply_rotation([0, 0, transform.get("Rz")])
                elif transform.get("T"):
                    transformation.apply_translation(transform.get("T"))

            material = object_data.get("material", {})
            color = material.get("Cs", np.array([0, 0, 0]))
            ka = material.get("Ka", 0.0)
            kd = material.get("Kd", 0.0)
            ks = material.get("Ks", 0.0)
            kt = material.get("Kt", 0.0)
            specularity = material.get("n", 1.0)
            texture_path = material.get("texture")
            texture = None

            if texture_path:
                texture = self.texture_manager.load_texture(texture_path)

            obj = Object(transformation, geometry_path, color, ka, kd, ks, kt, specularity, texture)
            self.objects.append(obj)

    def render(self):
        """
        Method to render the scene.

        Returns:
            (Image): Image of the rendered scene.

        """
        # Creating a new image.
        image = Image.new("RGB", self.camera.resolution, 0x000000)
        for y in range(self.camera.resolution[1]):
            for x in range(self.camera.resolution[0]):
                image.putpixel((x, y), (128, 128, 128))

        # TODO: Use G-buffer instead.
        z_buffer = np.matrix(np.ones((self.camera.resolution[0], self.camera.resolution[1])) * np.inf)

        for obj in self.objects:
            # Pushing the object transformation onto the stack and getting the concatenated matrix.
            self.transform_stack.push(obj.transformation.matrix)
            mvp_matrix = self.transform_stack.top()
            normal_transform_matrix = obj.transformation.get_inverse_transpose()

            for triangle in obj.geometry:
                v0 = triangle.v0
                v1 = triangle.v1
                v2 = triangle.v2

                # Setting up arrays
                pos0 = [[v0.pos[0]], [v0.pos[1]], [v0.pos[2]], [1]]
                pos1 = [[v1.pos[0]], [v1.pos[1]], [v1.pos[2]], [1]]
                pos2 = [[v2.pos[0]], [v2.pos[1]], [v2.pos[2]], [1]]
                n0 = [[v0.normal[0]], [v0.normal[1]], [v0.normal[2]], [0]]
                n1 = [[v1.normal[0]], [v1.normal[1]], [v1.normal[2]], [0]]
                n2 = [[v2.normal[0]], [v2.normal[1]], [v2.normal[2]], [0]]
                uv0 = np.array(v0.uv)
                uv1 = np.array(v1.uv)
                uv2 = np.array(v2.uv)

                # Applying vertex and normal transformations.
                pos0 = np.matmul(mvp_matrix, pos0)
                pos1 = np.matmul(mvp_matrix, pos1)
                pos2 = np.matmul(mvp_matrix, pos2)
                n0 = np.matmul(normal_transform_matrix, n0)
                n1 = np.matmul(normal_transform_matrix, n1)
                n2 = np.matmul(normal_transform_matrix, n2)

                n0 = np.transpose(n0[:-1])[0]
                n1 = np.transpose(n1[:-1])[0]
                n2 = np.transpose(n2[:-1])[0]

                n0 = n0 / np.sqrt(np.dot(n0, n0))
                n1 = n1 / np.sqrt(np.dot(n1, n1))
                n2 = n2 / np.sqrt(np.dot(n2, n2))

                # Homogenizing all the vectors.
                pos0 = pos0 / pos0[3]
                pos1 = pos1 / pos1[3]
                pos2 = pos2 / pos2[3]

                # Converting vertices to raster space.
                x0 = ((pos0[0] + 1) * ((self.camera.resolution[0] - 1) / 2))
                y0 = ((pos0[1] + 1) * ((self.camera.resolution[1] - 1) / 2))
                z0 = pos0[2]
                x1 = ((pos1[0] + 1) * ((self.camera.resolution[0] - 1) / 2))
                y1 = ((pos1[1] + 1) * ((self.camera.resolution[1] - 1) / 2))
                z1 = pos1[2]
                x2 = ((pos2[0] + 1) * ((self.camera.resolution[0] - 1) / 2))
                y2 = ((pos2[1] + 1) * ((self.camera.resolution[1] - 1) / 2))
                z2 = pos2[2]

                # Passing the transformed geometry to the fragment shader.
                fragment_shader(image, z_buffer, obj, self.camera, self.lights, (x0, y0, z0), (x1, y1, z1),
                                (x2, y2, z2), n0, n1, n2, uv0, uv1, uv2)

            # Popping the object transformation off the stack.
            self.transform_stack.pop()

        return image


if __name__ == "__main__":
    renderer = Renderer("table_scene.json")
    image = renderer.render()
    image.show()
