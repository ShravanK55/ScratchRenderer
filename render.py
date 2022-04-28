"""
Module implementing classes and methods related to rendering.
"""

from ambient_occlusion import SSAOKernel, SSAONoise
from argparse import ArgumentParser
from constants import MAX_RGB
from copy import deepcopy
from entity import Camera, Light, Object
from geometry import Transformation, TransformStack, mat_inverse_transpose
import json
import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image
from shader import geometry_pass_shader, lighting_pass_shader, occlusion_pass_shader, occlusion_blur_shader, \
    shadow_buffer_shader, wireframe_shader
from texture import TextureManager
from utils import is_backface, save_image_to_ppm


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
        self.resolution = resolution
        self.position_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.normal_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.color_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.material_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.specular_buffer = np.matrix(np.zeros((self.resolution[0], self.resolution[1])))
        self.depth_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])) * np.inf)
        self.occlusion_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))
        self.occlusion_blur_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))
        self.stencil_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))

        # Max depth and specularity are used for visualization purposes only.
        self.max_specularity = np.NINF
        self.max_depth = np.NINF

    def clear(self):
        """
        Method to clear the geometry buffer.
        """
        self.position_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.normal_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.color_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.material_buffer = [[[0, 0, 0] for _ in range(self.resolution[1])] for _ in range(self.resolution[0])]
        self.specular_buffer = np.matrix(np.zeros((self.resolution[0], self.resolution[1])))
        self.depth_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])) * np.inf)
        self.occlusion_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))
        self.occlusion_blur_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))
        self.stencil_buffer = np.matrix(np.ones((self.resolution[0], self.resolution[1])))

        # Max depth and specularity are used for visualization purposes only.
        self.max_specularity = np.NINF
        self.max_depth = np.NINF

    def get_attributes(self, x, y):
        """
        Method to get the attributes at a point in the geometry buffer.

        Args:
            x(int): X position in the geometry buffer.
            y(int): Y position in the geometry buffer.

        Returns:
            (tuple): Tuple containing a position vector, normal vector, color vector, material vector, specular value,
                depth value, occlusion value, occlusion blur value and a stencil value from the geometry buffer.

        """
        return (self.get_position(x, y), self.get_normal(x, y), self.get_color(x, y), self.get_material(x, y),
                self.get_specularity(x, y), self.get_depth(x, y), self.get_occlusion(x, y),
                self.get_occlusion_blur(x, y), self.get_stencil(x, y))

    def set_attributes(self, x, y, position=None, normal=None, color=None, material=None, specularity=0.0,
                       depth=np.inf, stencil=1):
        """
        Method to set the attributes at a point in the geometry buffer.

        Args:
            x(int): X position in the geometry buffer.
            y(int): Y position in the geometry buffer.
            position(list): Position vector. Defaults to None.
            normal(list): Normal vector. Defaults to None.
            color(list): Color vector. Defaults to None.
            material(list): Material vector. Defaults to None.
            specularity(float): Specularity value. Defaults to 0.0.
            depth(float): Depth value. Defaults to np.inf.
            stencil(float): Stencil value. Defaults to 1.

        Returns:
            (bool): Whether the attributes were updated in the geometry buffer.

        """
        success = self.set_depth(x, y, depth)
        if success:
            if position is not None:
                self.set_position(x, y, position)

            if normal is not None:
                self.set_normal(x, y, normal)

            if color is not None:
                self.set_color(x, y, color)

            if material is not None:
                self.set_material(x, y, material)

            self.set_specularity(x, y, specularity)
            self.set_stencil(x, y, stencil)

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

    def get_material(self, x, y):
        """
        Method to get a material on the material buffer.

        Args:
            x(int): X position in the material buffer.
            y(int): Y position in the material buffer.

        Returns:
            (list): Color from the color buffer.

        """
        return self.material_buffer[x][y]

    def set_material(self, x, y, material):
        """
        Method to set a material on the material buffer.

        Args:
            x(int): X position in the material buffer.
            y(int): Y position in the material buffer.
            material(list): Material vector.

        """
        self.material_buffer[x][y] = material

    def get_specularity(self, x, y):
        """
        Method to get a specular value on the specular buffer.

        Args:
            x(int): X position in the specular buffer.
            y(int): Y position in the specular buffer.

        Returns:
            (float): Specular value from the specular buffer.

        """
        return self.specular_buffer[x, y]

    def set_specularity(self, x, y, specularity):
        """
        Method to set a specular value on the specular buffer.

        Args:
            x(int): X position in the specular buffer.
            y(int): Y position in the specular buffer.
            specularity(float): Specular value.

        """
        self.specular_buffer[x, y] = specularity
        if specularity > self.max_specularity:
            self.max_specularity = specularity

    def get_depth(self, x, y):
        """
        Method to get a depth value on the depth buffer.

        Args:
            x(int): X position in the depth buffer.
            y(int): Y position in the depth buffer.

        Returns:
            (float): Depth value from the depth buffer.

        """
        return self.depth_buffer[x, y]

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
        if depth < self.depth_buffer[x, y]:
            self.depth_buffer[x, y] = depth
            if depth > self.max_depth:
                self.max_depth = depth
            return True

        return False

    def get_occlusion(self, x, y):
        """
        Method to get a occlusion value on the occlusion buffer.

        Args:
            x(int): X position in the occlusion buffer.
            y(int): Y position in the occlusion buffer.

        Returns:
            (float): Occlusion value from the occlusion buffer.

        """
        return self.occlusion_buffer[x, y]

    def set_occlusion(self, x, y, occlusion):
        """
        Method to set a occlusion value on the occlusion buffer.

        Args:
            x(int): X position in the occlusion buffer.
            y(int): Y position in the occlusion buffer.
            occlusion(float): Occlusion value.

        """
        self.occlusion_buffer[x, y] = occlusion

    def get_occlusion_blur(self, x, y):
        """
        Method to get a occlusion value on the occlusion blur buffer.

        Args:
            x(int): X position in the occlusion blur buffer.
            y(int): Y position in the occlusion blur buffer.

        Returns:
            (float): Occlusion value from the occlusion blur buffer.

        """
        return self.occlusion_blur_buffer[x, y]

    def set_occlusion_blur(self, x, y, occlusion_blur):
        """
        Method to set a occlusion value on the occlusion blur buffer.

        Args:
            x(int): X position in the occlusion blur buffer.
            y(int): Y position in the occlusion blur buffer.
            occlusion(float): Occlusion blur value.

        """
        self.occlusion_blur_buffer[x, y] = occlusion_blur

    def get_stencil(self, x, y):
        """
        Method to get a stencil value on the stencil buffer.

        Args:
            x(int): X position in the stencil buffer.
            y(int): Y position in the stencil buffer.

        Returns:
            (float): Stencil value from the stencil buffer.

        """
        return self.stencil_buffer[x, y]

    def set_stencil(self, x, y, stencil):
        """
        Method to set a stencil value on the stencil buffer.

        Args:
            x(int): X position in the stencil buffer.
            y(int): Y position in the stencil buffer.
            stencil(float): Stencil value.

        """
        self.stencil_buffer[x, y] = stencil


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
        self.shadow_bias = 0.015
        self.ssao_kernel = SSAOKernel(16)
        self.ssao_noise = SSAONoise(4)
        self.ssao_radius = 0.5
        self.ssao_bias = 0.025
        self.perlin_noise = PerlinNoise()
        self.cs_mid = 0.5
        self.cs_shadow = 0.15
        self.outline_max = 0.1
        self.outline_min = 0.04

        if scene_file_path:
            self.load_scene(scene_file_path)

    def load_scene(self, scene_file_path):
        """
        Method to load a scene to the renderer.

        Args:
            scene_file_path(str): File path of the scene. Defaults to None.

        """
        with open(scene_file_path) as json_file:
            json_data = json.load(json_file)
            renderer_params = json_data.get("renderer")
            self.scene = json_data.get("scene")

        # Loading renderer data.
        if renderer_params:
            self.shadow_bias = renderer_params.get("shadow_bias", self.shadow_bias)
            ssao_kernel_size = renderer_params.get("ssao_kernel_size")
            ssao_tile_size = renderer_params.get("ssao_tile_size")

            self.ssao_radius = renderer_params.get("ssao_radius", self.ssao_radius)
            self.ssao_bias = renderer_params.get("ssao_bias", self.ssao_bias)
            self.cs_mid = renderer_params.get("cs_mid", self.cs_mid)
            self.cs_shadow = renderer_params.get("cs_shadow", self.cs_shadow)
            self.outline_max = renderer_params.get("outline_max", self.outline_max)
            self.outline_min = renderer_params.get("outline_min", self.outline_min)

            if ssao_kernel_size:
                self.ssao_kernel = SSAOKernel(ssao_kernel_size)

            if ssao_tile_size:
                self.ssao_noise = SSAONoise(ssao_tile_size)

        # Getting camera data
        camera_data = self.scene.get("camera")
        camera_position = camera_data.get("from")
        camera_look_at = camera_data.get("to")
        camera_bounds = camera_data.get("bounds")
        camera_resolution = camera_data.get("resolution")
        camera_background_path = camera_data.get("background")
        camera_background_color = camera_data.get("background_color")
        camera_background = None
        camera_direction = [camera_position[i] - camera_look_at[i] for i in range(3)]
        camera_direction = camera_direction / np.sqrt(np.dot(camera_direction, camera_direction))

        if camera_background_path:
            camera_background = self.texture_manager.load_texture(camera_background_path)

        self.camera = Camera(camera_position, camera_direction, camera_bounds, camera_resolution, camera_background,
                             camera_background_color)
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

            if light_direction is not None:
                light.set_view_space_direction(self.camera.cam_matrix)

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
            kt = material.get("Kt", 1.0)
            specularity = material.get("n", 1.0)
            texture_path = material.get("texture")
            normal_map_path = material.get("normal_map")
            specular_map_path = material.get("specular_map")
            roughness_map_path = material.get("roughness_map")
            texture = None
            normal_map = None
            specular_map = None
            roughness_map = None

            if texture_path:
                texture = self.texture_manager.load_texture(texture_path)

            if normal_map_path:
                normal_map = self.texture_manager.load_texture(normal_map_path)

            if specular_map_path:
                specular_map = self.texture_manager.load_texture(specular_map_path)

            if roughness_map_path:
                roughness_map = self.texture_manager.load_texture(roughness_map_path)

            obj = Object(transformation, geometry_path, color, ka, kd, ks, kt, specularity, texture, normal_map,
                         specular_map, roughness_map)
            self.objects.append(obj)

    def render(self, enable_shadows=True, enable_ao=True, ndc_shift=None, cel_shade=False,
               halftone_shade=False, wireframe=False, line_art=False):
        """
        Method to render the scene.

        Args:
            enable_shadows(bool): Whether to render shadows. Defaults to True.
            enable_ao(bool): Whether to enable ambient occlusion. Defaults to True.
            ndc_shift(list): NDC shift to apply for anti aliasing. Defaults to None.
            cel_shade(bool): Whether to perform cel shading for the fragment. Defaults to False.
            halftone_shade(bool): Whether to perform halftone shading on the fragment. Defaults to False.
            wireframe(bool): Whether to render the scene as a wireframe. Defaults to False.
            line_art(bool): Whether to render the scene as line art. Defaults to False.

        Returns:
            (Image): Image of the rendered scene.

        """
        # Creating a new image.
        print("Creating a new render.")
        image = Image.new("RGB", self.camera.resolution, 0x000000)
        for y in range(self.camera.resolution[1]):
            for x in range(self.camera.resolution[0]):
                bg_color = (128, 128, 128)
                if self.camera.background_color is not None:
                    bg_color = self.camera.background_color

                image.putpixel((x, y), bg_color)

        if self.camera.background and not (halftone_shade or (line_art and not cel_shade) or wireframe):
            image.paste(self.camera.background, (0, 0))

        # Clearing all buffers.
        self.geometry_buffer.clear()
        for light in self.lights:
            if light.type != "ambient":
                light.clear_shadow_buffer()

        # Seed for Perlin noise.
        seed = np.random.uniform(0.0, 1.0)

        # Creating a shadow buffer for every light in the scene.
        if not wireframe and enable_shadows:
            print("Creating the shadow buffer.")
            shadow_buffer_shader(self.objects, self.lights)

        print("Geometry pass.")
        for obj in self.objects:
            # Pushing the object transformation onto the stack and getting the concatenated matrix.
            self.transform_stack.push(obj.transformation.matrix)
            mvp_matrix = self.transform_stack.top()
            mv_matrix = np.matmul(self.camera.cam_matrix, obj.transformation.matrix)
            normal_transform_matrix = mat_inverse_transpose(mv_matrix)

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

                # Getting the tangents at each vertex if it has a normal map.
                t0 = None
                t1 = None
                t2 = None
                if obj.normal_map:
                    delta_pos_1 = np.array([v1.pos[0] - v0.pos[0], v1.pos[1] - v0.pos[1], v1.pos[2] - v0.pos[2]])
                    delta_pos_2 = np.array([v2.pos[0] - v0.pos[0], v2.pos[1] - v0.pos[1], v2.pos[2] - v0.pos[2]])
                    delta_uv_1 = np.array([v1.uv[0] - v0.uv[0], v1.uv[1] - v0.uv[1]])
                    delta_uv_2 = np.array([v2.uv[0] - v0.uv[0], v2.uv[1] - v0.uv[1]])
                    r = 1.0 / (delta_uv_1[0] * delta_uv_2[1] - delta_uv_1[1] * delta_uv_2[0])
                    tangent = (delta_pos_1 * delta_uv_2[1] - delta_pos_2 * delta_uv_1[1]) * r
                    tangent = tangent / np.sqrt(np.dot(tangent, tangent))

                    if not (np.isnan(tangent[0]) or np.isnan(tangent[1]) or np.isnan(tangent[2])):
                        # Use the Gram-Schmidt process to orthogonalize the tangent to the normals.
                        nt0 = np.array([v0.normal[0], v0.normal[1], v0.normal[2]])
                        nt1 = np.array([v1.normal[0], v1.normal[1], v1.normal[2]])
                        nt2 = np.array([v2.normal[0], v2.normal[1], v2.normal[2]])
                        t0 = tangent - nt0 * np.dot(tangent, nt0)
                        t1 = tangent - nt1 * np.dot(tangent, nt1)
                        t2 = tangent - nt2 * np.dot(tangent, nt2)
                        t0 = t0 / np.sqrt(np.dot(t0, t0))
                        t1 = t1 / np.sqrt(np.dot(t1, t1))
                        t2 = t2 / np.sqrt(np.dot(t2, t2))
                        t0 = [[t0[0]], [t0[1]], [t0[2]], [0]]
                        t1 = [[t1[0]], [t1[1]], [t1[2]], [0]]
                        t2 = [[t2[0]], [t2[1]], [t2[2]], [0]]

                # Displacing backfaces for line art/outlines.
                backface = False
                if line_art:
                    w_pos = np.matmul(obj.transformation.matrix, pos0)
                    w_pos = np.transpose(w_pos[:-1])[0]

                    w_normal0 = np.matmul(obj.transformation.matrix, n0)
                    w_normal1 = np.matmul(obj.transformation.matrix, n1)
                    w_normal2 = np.matmul(obj.transformation.matrix, n2)
                    w_normal0 = np.transpose(w_normal0[:-1])[0]
                    w_normal1 = np.transpose(w_normal1[:-1])[0]
                    w_normal2 = np.transpose(w_normal2[:-1])[0]
                    w_normal0 = np.array(w_normal0 / np.sqrt(np.dot(w_normal0, w_normal0)))
                    w_normal1 = np.array(w_normal1 / np.sqrt(np.dot(w_normal1, w_normal1)))
                    w_normal2 = np.array(w_normal2 / np.sqrt(np.dot(w_normal2, w_normal2)))

                    p = np.array(w_pos)
                    n = (w_normal0 + w_normal1 + w_normal2) / 3
                    backface = is_backface(p, n, self.camera)

                    if backface:
                        outline_size = -self.perlin_noise(seed)
                        seed += 0.001
                        outline_size = self.outline_min + (self.outline_max - self.outline_min) * (outline_size)

                        pos0 = [[v0.pos[0] + v0.normal[0] * outline_size], [v0.pos[1] + v0.normal[1] * outline_size],
                                [v0.pos[2] + v0.normal[2] * outline_size], [1]]
                        pos1 = [[v1.pos[0] + v1.normal[0] * outline_size], [v1.pos[1] + v1.normal[1] * outline_size],
                                [v1.pos[2] + v1.normal[2] * outline_size], [1]]
                        pos2 = [[v2.pos[0] + v2.normal[0] * outline_size], [v2.pos[1] + v2.normal[1] * outline_size],
                                [v2.pos[2] + v2.normal[2] * outline_size], [1]]

                # Applying vertex and normal transformations.
                vpos0 = np.matmul(mv_matrix, pos0)
                vpos1 = np.matmul(mv_matrix, pos1)
                vpos2 = np.matmul(mv_matrix, pos2)
                pos0 = np.matmul(mvp_matrix, pos0)
                pos1 = np.matmul(mvp_matrix, pos1)
                pos2 = np.matmul(mvp_matrix, pos2)
                n0 = np.matmul(normal_transform_matrix, n0)
                n1 = np.matmul(normal_transform_matrix, n1)
                n2 = np.matmul(normal_transform_matrix, n2)

                if obj.normal_map and t0 is not None and t1 is not None and t2 is not None:
                    t0 = np.matmul(normal_transform_matrix, t0)
                    t1 = np.matmul(normal_transform_matrix, t1)
                    t2 = np.matmul(normal_transform_matrix, t2)

                # Conversion to row vectors and normalization.
                vpos0 = np.transpose(vpos0)[0]
                vpos1 = np.transpose(vpos1)[0]
                vpos2 = np.transpose(vpos2)[0]
                pos0 = np.transpose(pos0)[0]
                pos1 = np.transpose(pos1)[0]
                pos2 = np.transpose(pos2)[0]
                n0 = np.transpose(n0[:-1])[0]
                n1 = np.transpose(n1[:-1])[0]
                n2 = np.transpose(n2[:-1])[0]
                n0 = n0 / np.sqrt(np.dot(n0, n0))
                n1 = n1 / np.sqrt(np.dot(n1, n1))
                n2 = n2 / np.sqrt(np.dot(n2, n2))
                if obj.normal_map and t0 is not None and t1 is not None and t2 is not None:
                    t0 = np.transpose(t0[:-1])[0]
                    t1 = np.transpose(t1[:-1])[0]
                    t2 = np.transpose(t2[:-1])[0]
                    t0 = t0 / np.sqrt(np.dot(t0, t0))
                    t1 = t1 / np.sqrt(np.dot(t1, t1))
                    t2 = t2 / np.sqrt(np.dot(t2, t2))

                # Homogenizing all the vectors.
                pos0 = pos0 / pos0[3]
                pos1 = pos1 / pos1[3]
                pos2 = pos2 / pos2[3]

                # Adding NDC shift for anti aliasing.
                if ndc_shift is not None:
                    pos0 = [pos0[0] + (ndc_shift[0] / (self.camera.resolution[0] - 1)),
                            pos0[1] + (ndc_shift[0] / (self.camera.resolution[1] - 1)),
                            pos0[2], pos0[3]]
                    pos1 = [pos1[0] + (ndc_shift[0] / (self.camera.resolution[0] - 1)),
                            pos1[1] + (ndc_shift[0] / (self.camera.resolution[1] - 1)),
                            pos1[2], pos1[3]]
                    pos2 = [pos2[0] + (ndc_shift[0] / (self.camera.resolution[0] - 1)),
                            pos2[1] + (ndc_shift[0] / (self.camera.resolution[1] - 1)),
                            pos2[2], pos2[3]]

                # Converting vertices to raster space.
                pos0 = np.array([(pos0[0] + 1) * ((self.camera.resolution[0] - 1) / 2),
                                (pos0[1] + 1) * ((self.camera.resolution[1] - 1) / 2),
                                pos0[2]])
                pos1 = np.array([(pos1[0] + 1) * ((self.camera.resolution[0] - 1) / 2),
                                (pos1[1] + 1) * ((self.camera.resolution[1] - 1) / 2),
                                pos1[2]])
                pos2 = np.array([(pos2[0] + 1) * ((self.camera.resolution[0] - 1) / 2),
                                (pos2[1] + 1) * ((self.camera.resolution[1] - 1) / 2),
                                pos2[2]])

                if wireframe:
                    wireframe_shader(image, self.camera, pos0, pos1, pos2)
                    continue

                # Passing the geometry to the first pass of the deferred fragment shader.
                geometry_pass_shader(self.geometry_buffer, obj, self.camera, pos0, pos1, pos2, vpos0, vpos1, vpos2,
                                     n0, n1, n2, uv0, uv1, uv2, t0, t1, t2, line_art, backface)

            # Popping the object transformation off the stack.
            self.transform_stack.pop()

        if wireframe:
            return image

        if enable_ao:
            print("Occlusion pass.")
            # Calculating the occlusion values for ambient occlusion.
            occlusion_pass_shader(self.geometry_buffer, self.camera, self.ssao_kernel, self.ssao_noise,
                                  self.ssao_radius, self.ssao_bias)

            # Blurring the occlusion buffer to remove noise.
            occlusion_blur_shader(self.geometry_buffer, self.camera, self.ssao_noise)

        print("Lighting pass.")
        # Calculating the lighting in the second pass of the deferred fragment shader.
        lighting_pass_shader(image, self.geometry_buffer, self.camera, self.lights, cel_shade, halftone_shade, line_art,
                             self.shadow_bias, self.cs_mid, self.cs_shadow)

        print("Finished render.")
        return image

    def render_geometry_buffer(self):
        """
        Method to render a visualization of the geometry buffer.

        Returns:
            (tuple): A tuple of images containing visualizations of the position, normal, color, specularity, depth,
                occlusion and stencil buffers.

        """
        # Creating images for the buffers.
        position_image = Image.new("RGB", self.camera.resolution, 0x000000)
        normal_image = Image.new("RGB", self.camera.resolution, 0x000000)
        albedo_image = Image.new("RGB", self.camera.resolution, 0x000000)
        specular_image = Image.new("RGB", self.camera.resolution, 0x000000)
        depth_image = Image.new("RGB", self.camera.resolution, 0x000000)
        occlusion_noise_image = Image.new("RGB", self.camera.resolution, 0x000000)
        occlusion_image = Image.new("RGB", self.camera.resolution, 0x000000)
        stencil_image = Image.new("RGB", self.camera.resolution, 0x000000)
        for y in range(self.camera.resolution[1]):
            for x in range(self.camera.resolution[0]):
                position_image.putpixel((x, y), (0, 0, 0))
                normal_image.putpixel((x, y), (0, 0, 0))
                albedo_image.putpixel((x, y), (0, 0, 0))
                specular_image.putpixel((x, y), (0, 0, 0))
                depth_image.putpixel((x, y), (0, 0, 0))
                occlusion_noise_image.putpixel((x, y), (0, 0, 0))
                occlusion_image.putpixel((x, y), (0, 0, 0))
                stencil_image.putpixel((x, y), (0, 0, 0))

        print("Rendering geometry buffers.")
        for y in range(self.camera.resolution[1]):
            for x in range(self.camera.resolution[0]):
                position, normal, color, _, specularity, depth, occlusion, occlusion_blur, stencil = \
                    self.geometry_buffer.get_attributes(x, y)

                if depth != np.inf:
                    # Adding the position value color.
                    v_pos = deepcopy(position)
                    v_pos = v_pos / np.sqrt(np.dot(v_pos, v_pos))
                    pos_color = (round(abs(v_pos[0]) * MAX_RGB), round(abs(v_pos[1]) * MAX_RGB),
                                 round(abs(v_pos[2]) * MAX_RGB))
                    position_image.putpixel((x, self.camera.resolution[1] - y - 1), pos_color)

                    # Adding the normal value color.
                    normal_color = (round(normal[0] * MAX_RGB), round(normal[1] * MAX_RGB), round(normal[2] * MAX_RGB))
                    normal_image.putpixel((x, self.camera.resolution[1] - y - 1), normal_color)

                    # Adding the albedo color.
                    albedo_color = (round(color[0] * MAX_RGB), round(color[1] * MAX_RGB), round(color[2] * MAX_RGB))
                    albedo_image.putpixel((x, self.camera.resolution[1] - y - 1), albedo_color)

                    # Adding the specular color.
                    max_specularity = self.geometry_buffer.max_specularity if \
                        self.geometry_buffer.max_specularity != 0 else 1.0
                    specular_val = round(specularity / max_specularity * MAX_RGB)
                    specular_color = (specular_val, specular_val, specular_val)
                    specular_image.putpixel((x, self.camera.resolution[1] - y - 1), specular_color)

                    # Adding the depth color.
                    max_depth = self.geometry_buffer.max_depth if self.geometry_buffer.max_depth != 0 else 1.0
                    depth_val = round(depth / max_depth * MAX_RGB)
                    depth_color = (depth_val, depth_val, depth_val)
                    depth_image.putpixel((x, self.camera.resolution[1] - y - 1), depth_color)

                    # Adding the occlusion color.
                    occlusion_noise_val = round(occlusion * MAX_RGB)
                    occlusion_noise_color = (occlusion_noise_val, occlusion_noise_val, occlusion_noise_val)
                    occlusion_noise_image.putpixel((x, self.camera.resolution[1] - y - 1), occlusion_noise_color)

                    # Adding the occlusion color.
                    occlusion_val = round(occlusion_blur * MAX_RGB)
                    occlusion_color = (occlusion_val, occlusion_val, occlusion_val)
                    occlusion_image.putpixel((x, self.camera.resolution[1] - y - 1), occlusion_color)

                    # Adding the stencil color.
                    stencil_val = round(stencil * MAX_RGB)
                    stencil_color = (stencil_val, stencil_val, stencil_val)
                    stencil_image.putpixel((x, self.camera.resolution[1] - y - 1), stencil_color)

        return (position_image, normal_image, albedo_image, specular_image, depth_image, occlusion_noise_image,
                occlusion_image, stencil_image)

    def render_shadow_buffers(self):
        """
        Method to render visualizations of the shadow buffers in every light.

        Returns:
            (list): A list of images containing visualizations of the shadow buffers of each light.

        """
        print("Rendering shadow buffers.")
        shadow_buffer_images = []

        for light in self.lights:
            if light.type == "ambient":
                continue

            # Creating images for the buffers.
            depth_image = Image.new("RGB", light.camera.resolution, 0x000000)
            for y in range(light.camera.resolution[1]):
                for x in range(light.camera.resolution[0]):
                    depth_image.putpixel((x, y), (0, 0, 0))

            for y in range(light.camera.resolution[1]):
                for x in range(light.camera.resolution[0]):
                    depth = light.get_shadow_buffer_depth(x, y)

                    if depth != np.inf:
                        # Adding the depth color.
                        depth_val = round(depth / light.max_depth * MAX_RGB)
                        depth_color = (depth_val, depth_val, depth_val)
                        depth_image.putpixel((x, self.camera.resolution[1] - y - 1), depth_color)

            shadow_buffer_images.append(depth_image)

        return shadow_buffer_images


if __name__ == "__main__":
    parser = ArgumentParser(description="A program that renders 3D scenes.")
    parser.add_argument("-s", "--scene", help="Scene file path.", type=str, default="scenes/jinx_scene.json")
    parser.add_argument("--disable_shadows", help="Disable shadows in the renderer.", action="store_true")
    parser.add_argument("--disable_ssao", help="Disable ambient occlusion in the renderer.", action="store_true")
    parser.add_argument("-a", "--anti_aliasing", help="Enable anti aliasing in the renderer.", action="store_true")
    parser.add_argument("-c", "--cel_shade", help="Turn on cel shading.", action="store_true")
    parser.add_argument("-t", "--halftone_shade", help="Turn on halftone shading.", action="store_true")
    parser.add_argument("-l", "--line_art", help="Turn on line art shading.", action="store_true")
    parser.add_argument("-w", "--wireframe", help="Turn on wireframe mode.", action="store_true")
    parser.add_argument("-d", "--draw_buffers", help="Draw the geometry and shadow buffers.", action="store_true")
    parser.add_argument("-f", "--write_to_file", help="Write the output images to files.", action="store_true")
    args = parser.parse_args()

    np.seterr(divide='ignore', invalid='ignore')
    renderer = Renderer(args.scene)
    AA_SHIFTS = [[-0.52, 0.38], [0.41, 0.56], [0.27, 0.08], [-0.17, -0.29], [0.58, -0.55], [-0.31, -0.71]]
    AA_WEIGHTS = [0.128, 0.119, 0.294, 0.249, 0.104, 0.106]
    ENABLE_SHADOWS = False if args.disable_shadows else True
    ENABLE_AO = False if args.disable_ssao else True
    USE_AA = args.anti_aliasing
    CEL_SHADE = args.cel_shade
    HALFTONE_SHADE = args.halftone_shade
    LINE_ART = args.line_art
    WIREFRAME = args.wireframe
    RENDER_AA_IMAGES = False
    RENDER_GEOMETRY_BUFFER = args.draw_buffers
    RENDER_SHADOW_BUFFERS = args.draw_buffers
    WRITE_TO_FILE = args.write_to_file
    WRITE_BUFFERS_TO_FILE = args.write_to_file
    OUTPUT_FILE_NAME = "render.ppm"
    POSITION_FILE_NAME = "position.ppm"
    NORMAL_FILE_NAME = "normal.ppm"
    ALBEDO_FILE_NAME = "albedo.ppm"
    SPECULAR_FILE_NAME = "specular.ppm"
    DEPTH_FILE_NAME = "depth.ppm"
    OCCLUSION_NOISE_FILE_NAME = "occlusion_noise.ppm"
    OCCLUSION_FILE_NAME = "occlusion.ppm"
    STENCIL_FILE_NAME = "stencil.ppm"
    SHADOW_BUFFER_FILE_NAME = "shadow{}.ppm"

    if USE_AA:
        aa_images = []
        for aa_shift in AA_SHIFTS:
            aa_image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, ndc_shift=aa_shift,
                                       cel_shade=CEL_SHADE, halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME,
                                       line_art=LINE_ART)
            aa_images.append(aa_image)

            if RENDER_AA_IMAGES:
                aa_image.show()

        image = Image.new("RGB", renderer.camera.resolution, 0x000000)
        for y in range(renderer.camera.resolution[1]):
            for x in range(renderer.camera.resolution[0]):
                out_color = np.array([0.0, 0.0, 0.0])

                for i in range(len(aa_images)):
                    color = np.array(aa_images[i].getpixel((x, y)))
                    out_color += [color[0] * AA_WEIGHTS[i], color[1] * AA_WEIGHTS[i], color[2] * AA_WEIGHTS[i]]

                out_color = (round(out_color[0]), round(out_color[1]), round(out_color[2]))
                image.putpixel((x, y), tuple(out_color))

        image.show()

        if WRITE_TO_FILE:
            save_image_to_ppm(image, OUTPUT_FILE_NAME)

    else:
        image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                                halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
        image.show()

        if WRITE_TO_FILE:
            print("Writing the render to {}.".format(OUTPUT_FILE_NAME))
            save_image_to_ppm(image, OUTPUT_FILE_NAME)

    if RENDER_GEOMETRY_BUFFER:
        position_image, normal_image, albedo_image, specular_image, depth_image, occlusion_noise_image, \
            occlusion_image, stencil_image = renderer.render_geometry_buffer()
        position_image.show()
        normal_image.show()
        albedo_image.show()
        specular_image.show()
        depth_image.show()
        occlusion_noise_image.show()
        occlusion_image.show()
        stencil_image.show()

        if WRITE_BUFFERS_TO_FILE:
            print("Writing the geometry buffers to files.")
            save_image_to_ppm(position_image, POSITION_FILE_NAME)
            save_image_to_ppm(normal_image, NORMAL_FILE_NAME)
            save_image_to_ppm(albedo_image, ALBEDO_FILE_NAME)
            save_image_to_ppm(specular_image, SPECULAR_FILE_NAME)
            save_image_to_ppm(depth_image, DEPTH_FILE_NAME)
            save_image_to_ppm(occlusion_noise_image, OCCLUSION_NOISE_FILE_NAME)
            save_image_to_ppm(occlusion_image, OCCLUSION_FILE_NAME)
            save_image_to_ppm(stencil_image, STENCIL_FILE_NAME)

    if RENDER_SHADOW_BUFFERS:
        shadow_buffer_images = renderer.render_shadow_buffers()
        shadow_buffer_idx = 0
        for shadow_buffer_image in shadow_buffer_images:
            shadow_buffer_image.show()

            if WRITE_BUFFERS_TO_FILE:
                print("Writing shadow buffers to files.")
                save_image_to_ppm(shadow_buffer_image, SHADOW_BUFFER_FILE_NAME.format(shadow_buffer_idx))

            shadow_buffer_idx += 1
