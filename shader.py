"""
Module implementing shader functions.
"""

from geometry import TransformStack, mat_inverse, mat_inverse_transpose
import math
import numpy as np
from texture import get_texture_color
from utils import triangle_area, smooth_step


# FORWARD RENDERING

def light_calc(obj, camera, lights, n, uv=None):
    """
    Method to calculate the color of a fragment from lighting.

    Args:
        obj(Object): Object being rendered.
        camera(Camera): Camera that is viewing the object.
        lights(list): List of lights in the scene.
        n(list): Normal vector of the fragment being shaded.
        uv(list): UV co-ordinates for texture mapping. Defaults to None.

    Returns:
        (list): Color of the fragment.

    """
    MAX_RGB = 255
    object_color = np.array(obj.color)
    ambient_color = np.array([0.0, 0.0, 0.0])
    diffuse_color = np.array([0.0, 0.0, 0.0])
    specular_color = np.array([0.0, 0.0, 0.0])

    for light in lights:
        light_color = np.array(light.color) * light.intensity

        if light.type == "ambient":
            ambient_color = light_color
            continue

        # Getting the light, normal, eye direction and reflected light vectors.
        l = light.direction
        e = camera.direction
        r = 2 * np.dot(n, l) * n - l
        r = r / np.sqrt(np.dot(r, r))

        # Calculating the N dot L, N dot E and R dot E products.
        n_dot_l = np.dot(n, l)
        n_dot_e = np.dot(n, e)
        r_dot_e = np.dot(r, e)

        # Clamping the value of R dot E between 0 and 1.
        r_dot_e = max(r_dot_e, 0)
        r_dot_e = min(r_dot_e, 1)

        # Adding to the specular intensity.
        specular_color += light_color * (r_dot_e ** obj.specularity)

        # Adding to diffuse color depending on N dot L and N dot E.
        if n_dot_l > 0 and n_dot_e > 0:
            diffuse_color += light_color * n_dot_l
        elif n_dot_l < 0 and n_dot_e < 0:
            diffuse_color += light_color * n_dot_l * -1.0

    # Calculating the final color.
    color = [0, 0, 0]
    if (uv is not None) and obj.texture:
        object_color = get_texture_color(obj.texture, uv) * MAX_RGB * obj.kt
        color = object_color * ((obj.ka * ambient_color) + (obj.kd * diffuse_color) + (obj.ks * specular_color))
        color = (round(color[0]), round(color[1]), round(color[2]))
    else:
        color = object_color * ((obj.ka * ambient_color) + (obj.kd * diffuse_color)) + (obj.ks * specular_color)
        color = (round(color[0] * MAX_RGB), round(color[1] * MAX_RGB), round(color[2] * MAX_RGB))

    return color


def fragment_shader(image, z_buffer, obj, camera, lights, pos0, pos1, pos2, n0, n1, n2, uv0, uv1, uv2):
    """
    Method implementing a fragment shader (forward rendering).

    Args:
        image(Image): Image to write the pixels into.
        z_buffer(matrix): Z buffer of the image being rendered.
        obj(Object): Object being rendered.
        camera(Camera): Camera that is viewing the scene.
        lights(list): List of lights in the scene.
        pos0(list): Position of vertex 0.
        pos1(list): Position of vertex 1.
        pos2(list): Position of vertex 2.
        n0(list): Normal of vertex 0.
        n1(list): Normal of vertex 0.
        n2(list): Normal of vertex 0.
        uv0(list): UV of vertex 0.
        uv1(list): UV of vertex 1.
        uv2(list): UV of vertex 2.

    """
    # Getting the co-ordinates from vectors.
    x0, y0, z0 = pos0
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    # Dividing UVs by their Z values for perspective correction.
    p_uv0 = uv0 / pos0[2]
    p_uv1 = uv1 / pos1[2]
    p_uv2 = uv2 / pos2[2]

    # Getting the bounding box of the triangle.
    xmin = max(math.floor(min(x0, x1, x2)), 0)
    xmax = min(math.ceil(max(x0, x1, x2)), camera.resolution[0])
    ymin = max(math.floor(min(y0, y1, y2)), 0)
    ymax = min(math.ceil(max(y0, y1, y2)), camera.resolution[1])

    # Skip scan conversion if the area covered by the triangle is 0.
    if triangle_area((x0, y0), (x1, y1), (x2, y2)) == 0:
        return

    # Scan converting the triangle.
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            # Getting the barycentric co-ordinates of the pixel.
            alpha = triangle_area((x, y), (x1, y1), (x2, y2)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
            beta = triangle_area((x, y), (x2, y2), (x0, y0)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
            gamma = triangle_area((x, y), (x0, y0), (x1, y1)) / triangle_area((x0, y0), (x1, y1), (x2, y2))

            # Interpolating the Z value.
            z = alpha * z0 + beta * z1 + gamma * z2

            if alpha >= 0 and beta >= 0 and gamma >= 0:
                # Perspective correcting the UVs.
                uv = alpha * p_uv0 + beta * p_uv1 + gamma * p_uv2
                z_at_pixel = 1 / (alpha * (1 / z0) + beta * (1 / z1) + gamma * (1 / z2))
                uv = uv * z_at_pixel

                # Shading the pixel.
                if z < z_buffer[x, y]:
                    n = alpha * n0 + beta * n1 + gamma * n2
                    color = light_calc(obj, camera, lights, n, uv)
                    image.putpixel((x, -y), color)
                    z_buffer[x, y] = z


# DEFERRED RENDERING

def shade_fragment(fragment_pos, lights, n, fragment_color, material, specularity, occlusion=1.0, shadow=0.0,
                   cel_shade=False):
    """
    Method to calculate the color of a fragment from lighting (Deferred lighting pass).

    Args:
        fragment_pos(list): Position of the fragment in view space.
        lights(list): List of lights in the scene.
        n(list): Normal vector of the fragment being shaded.
        fragment_color(list): Color of the fragment being rendered.
        material(list): Material of the fragment being rendered.
        specularity(float): Specularity of the fragment being rendered.
        occlusion(float): Occlusion of the fragment being rendered. Defaults to 1.0.
        shadow(float): Amount of shadow on the fragment being rendered. Defaults to 1.0.
        cel_shade(bool): Whether to perform cel shading on the fragment. Defaults to False.

    Returns:
        (list): Color of the fragment.

    """
    MAX_RGB = 255
    object_color = np.array(fragment_color)
    ambient_color = np.array([0.0, 0.0, 0.0])
    diffuse_color = np.array([0.0, 0.0, 0.0])
    specular_color = np.array([0.0, 0.0, 0.0])

    for light in lights:
        light_color = np.array(light.color) * light.intensity

        if light.type == "ambient":
            ambient_color = light_color
            continue

        # Getting the light, normal, eye direction and reflected light vectors.
        l = light.view_space_direction
        e = -1 * fragment_pos[:-1]
        e = e / np.sqrt(np.dot(e, e))
        r = 2 * np.dot(n, l) * n - l
        r = r / np.sqrt(np.dot(r, r))

        # Calculating the N dot L, N dot E and R dot E products.
        n_dot_l = np.dot(n, l)
        n_dot_e = np.dot(n, e)
        r_dot_e = np.dot(r, e)

        if cel_shade:
            light_scale = 0
            if n_dot_l <= 0.01:
                light_scale = 0
            elif (n_dot_l > 0.5 and n_dot_l <= 1):
                light_scale = 1
            else:
                light_scale = 0.5

            # Getting the halfway vec between the light direction and camera direction.
            half_vec = np.array(l) + np.array(e)
            half_vec = half_vec / np.sqrt(np.dot(half_vec, half_vec))
            n_dot_h = np.dot(n, half_vec)

            # Accumulating the diffuse and specular color.
            diffuse_color += light_color * light_scale
            specular_color += light_color * ((n_dot_h * light_scale) ** specularity)
            continue

        # Clamping the value of R dot E between 0 and 1.
        r_dot_e = max(r_dot_e, 0)
        r_dot_e = min(r_dot_e, 1)

        # Accumulating the specular color.
        specular_color += light_color * (r_dot_e ** specularity)

        # Accumulating the diffuse color.
        if n_dot_l > 0 and n_dot_e > 0:
            diffuse_color += light_color * n_dot_l
        elif n_dot_l < 0 and n_dot_e < 0:
            diffuse_color += light_color * n_dot_l * -1.0

    # Quantizing occlusion for cel shading.
    if cel_shade:
        if occlusion <= 0.01:
            occlusion = 0
        elif (occlusion > 0.5 and occlusion <= 1):
            occlusion = 1
        else:
            occlusion = 0.5

    # Calculating the final color.
    ka, kd, ks = material
    color = object_color * ((ka * ambient_color) + ((kd * diffuse_color) + (ks * specular_color)) * (1.0 - shadow)) * \
        occlusion
    color = (round(color[0] * MAX_RGB), round(color[1] * MAX_RGB), round(color[2] * MAX_RGB))

    return color


def geometry_pass_shader(g_buffer, obj, camera, pos0, pos1, pos2, vpos0, vpos1, vpos2, n0, n1, n2, uv0,
                         uv1, uv2):
    """
    Method implementing a geometry pass shader (First pass in deferred rendering).

    Args:
        g_buffer(GeometryBuffer): Geometry buffer of the image being rendered.
        obj(Object): Object being rendered.
        camera(Camera): Camera that is viewing the scene.
        pos0(list): Position of vertex 0.
        pos1(list): Position of vertex 1.
        pos2(list): Position of vertex 2.
        vpos0(list): View space position of vertex 0.
        vpos1(list): View space position of vertex 1.
        vpos2(list): View space position of vertex 2.
        n0(list): Normal of vertex 0.
        n1(list): Normal of vertex 0.
        n2(list): Normal of vertex 0.
        uv0(list): UV of vertex 0.
        uv1(list): UV of vertex 1.
        uv2(list): UV of vertex 2.

    """
    # Getting the co-ordinates from vectors.
    x0, y0, z0 = pos0
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    # Dividing UVs by their Z values for perspective correction.
    p_uv0 = uv0 / pos0[2]
    p_uv1 = uv1 / pos1[2]
    p_uv2 = uv2 / pos2[2]

    # Getting the bounding box of the triangle.
    xmin = max(math.floor(min(x0, x1, x2)), 0)
    xmax = min(math.ceil(max(x0, x1, x2)), camera.resolution[0])
    ymin = max(math.floor(min(y0, y1, y2)), 0)
    ymax = min(math.ceil(max(y0, y1, y2)), camera.resolution[1])

    # Skip scan conversion if the area covered by the triangle is 0.
    if triangle_area((x0, y0), (x1, y1), (x2, y2)) == 0:
        return

    # Scan converting the triangle.
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            # Getting the barycentric co-ordinates of the pixel.
            alpha = triangle_area((x, y), (x1, y1), (x2, y2)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
            beta = triangle_area((x, y), (x2, y2), (x0, y0)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
            gamma = triangle_area((x, y), (x0, y0), (x1, y1)) / triangle_area((x0, y0), (x1, y1), (x2, y2))

            if alpha >= 0 and beta >= 0 and gamma >= 0:
                # Barycentrically interpolating the view space position vectors.
                vpos = alpha * vpos0 + beta * vpos1 + gamma * vpos2

                # Getting the interpolated normal.
                n = alpha * n0 + beta * n1 + gamma * n2

                # Perspective correcting the UVs.
                uv = alpha * p_uv0 + beta * p_uv1 + gamma * p_uv2
                z_at_pixel = 1 / (alpha * (1 / z0) + beta * (1 / z1) + gamma * (1 / z2))
                uv = uv * z_at_pixel

                # Getting the object/texture color.
                color = obj.color
                if obj.texture:
                    color = get_texture_color(obj.texture, uv) * obj.kt

                # Getting the object material.
                material = [obj.ka, obj.kd, obj.ks]

                # Interpolating the Z value.
                z = alpha * z0 + beta * z1 + gamma * z2

                # Adding geometry to the geometry buffer.
                g_buffer.set_attributes(x, y, vpos, n, color, material, obj.specularity, z)


def occlusion_pass_shader(g_buffer, camera, kernel, noise, radius=0.5, bias=0.025):
    """
    Method to calculate the occlusion of all the fragments in the scene.

    Args:
        g_buffer(GeometryBuffer): Geometry buffer of the image being rendered.
        camera(Camera): Camera that is viewing the scene.
        kernel(SSAOKernel): SSAO kernel to use for sampling the scene.
        noise(SSAONoise): SSAO noise to use for sampling the scene.
        radius(float): Radius of the sample hemisphere. Defaults to 0.5.
        bias(float): Bias to use when determining whether a sample point is occluded or not. Defaults to 0.025.

    """
    projection_matrix = camera.projection_matrix

    for y in range(camera.resolution[1]):
        for x in range(camera.resolution[0]):
            position, normal, _, _, _, depth, _, _ = g_buffer.get_attributes(x, y)

            if depth != np.inf:
                # Getting a random basis to form the sampling hemisphere around.
                random_vec = noise.sample(x, y)
                tangent = random_vec - normal * np.dot(random_vec, normal)
                bitangent = np.cross(normal, tangent)
                tbn_matrix = [[tangent[0], bitangent[0], normal[0]],
                              [tangent[1], bitangent[1], normal[1]],
                              [tangent[2], bitangent[2], normal[2]]]

                occlusion = 0.0
                for sample in kernel.samples:
                    # Getting a random sample point in view space.
                    sample_vector = [[sample[0]], [sample[1]], [sample[2]]]
                    sample_pos = np.matmul(tbn_matrix, sample_vector)
                    sample_pos = np.transpose(sample_pos)[0]
                    sample_pos = position[:-1] + sample_pos * radius

                    # Projection of position to screen space to get sampling location on position texture.
                    offset = [[sample_pos[0]], [sample_pos[1]], [sample_pos[2]], [1]]
                    offset = np.matmul(projection_matrix, offset)
                    offset = np.transpose(offset)[0]
                    offset = offset / offset[3]
                    offset = (offset + 1) / 2

                    # Converting to raster space for sampling.
                    texture_pos_x = min(offset[0] * (camera.resolution[0] - 1), camera.resolution[0] - 1)
                    texture_pos_x = max(texture_pos_x, 0)
                    texture_pos_y = min(offset[1] * (camera.resolution[1] - 1), camera.resolution[1] - 1)
                    texture_pos_y = max(texture_pos_y, 0)

                    # Gettting the position from the position buffer.
                    sample_v_pos = g_buffer.get_position(int(texture_pos_x), int(texture_pos_y))
                    sample_depth = sample_v_pos[2]

                    # Occlusion accumulation based on range check.
                    range_check = smooth_step(0.0, 1.0, radius / np.abs(position[2] - sample_depth))
                    if sample_depth >= position[2] + bias:
                        occlusion += range_check

                occlusion = 1.0 - (occlusion / kernel.size)
                g_buffer.set_occlusion(x, y, occlusion)


def occlusion_blur_shader(g_buffer, camera, noise):
    """
    Method to blur the occlusion buffer to remove noise.

    Args:
        g_buffer(GeometryBuffer): Geometry buffer of the image being rendered.
        camera(Camera): Camera that is viewing the scene.
        noise(SSAONoise): SSAO noise to use for getting blur parameters.

    """
    tile_size = noise.tile_size
    for y in range(camera.resolution[1]):
        for x in range(camera.resolution[0]):
            _, _, _, _, _, depth, _, _ = g_buffer.get_attributes(x, y)

            if depth != np.inf:
                result = 0.0
                num_accumulated = 0

                # Blurring the noise based on tile size.
                for oy in range(-int(tile_size / 2), int(tile_size / 2)):
                    for ox in range(-int(tile_size / 2), int(tile_size / 2)):
                        offset_x = x + ox
                        offset_y = y + oy

                        # Clamping offsets to the texture bounds.
                        offset_x = min(offset_x, camera.resolution[0] - 1)
                        offset_x = max(offset_x, 0)
                        offset_y = min(offset_y, camera.resolution[1] - 1)
                        offset_y = max(offset_y, 0)

                        _, _, _, _, _, offset_depth, occlusion, _ = g_buffer.get_attributes(offset_x, offset_y)

                        if offset_depth != np.inf:
                            result += occlusion
                            num_accumulated += 1

                occlusion_blur = result / num_accumulated
                g_buffer.set_occlusion_blur(x, y, occlusion_blur)


def lighting_pass_shader(image, g_buffer, camera, lights, cel_shade=False, shadow_bias=0.005):
    """
    Method implementing a lighting pass shader (Second pass in deferred rendering).

    Args:
        image(Image): Image to write the pixels into.
        g_buffer(GeometryBuffer): Geometry buffer of the image being rendered.
        camera(Camera): Camera that is viewing the scene.
        lights(list): List of lights in the scene.
        cel_shade(bool): Whether to perform cel shading for the fragment. Defaults to False.
        shadow_bias(float): Bias to use when checking whether a fragment is in shadow. Defaults to 0.005.

    """
    transform_stack = TransformStack()
    transform_stack.push(camera.projection_matrix)
    transform_stack.push(camera.cam_matrix)
    inv_vp_matrix = np.linalg.inv(transform_stack.top())
    inv_cam_matrix = mat_inverse(mat_inverse_transpose(camera.cam_matrix))

    for y in range(camera.resolution[1]):
        for x in range(camera.resolution[0]):
            position, normal, color, material, specularity, depth, _, occlusion_blur = g_buffer.get_attributes(x, y)

            if depth != np.inf:
                n = [normal[0], normal[1], normal[2], 0]

                # Calculating the shadow value of a fragment.
                ndc_x = (x / ((camera.resolution[0] - 1) / 2)) - 1
                ndc_y = (y / ((camera.resolution[1] - 1) / 2)) - 1
                w_pos = np.matmul(inv_vp_matrix, [ndc_x, ndc_y, depth, 1])
                w_normal = np.matmul(inv_cam_matrix, n)
                w_normal = w_normal / np.sqrt(np.dot(w_normal, w_normal))
                shadow = get_fragment_shadow(w_pos, w_normal, lights, shadow_bias)

                # Calculating the final color of a fragment from lighting.
                color = shade_fragment(position, lights, normal, color, material, specularity, occlusion_blur, shadow,
                                       cel_shade)
                image.putpixel((x, -y), color)


def shadow_buffer_shader(objects, lights):
    """
    Method to render the shadow buffer in all the lights of a scene.

    Args:
        objects(list): List of objects in the scene.
        lights(list): List of lights in the scene.

    """
    for light in lights:
        if light.type == "ambient":
            continue

        light.clear_shadow_buffer()
        transform_stack = TransformStack()
        transform_stack.push(light.camera.projection_matrix)
        transform_stack.push(light.camera.cam_matrix)

        for obj in objects:
            # Pushing the object transformation onto the stack and getting the concatenated matrix.
            transform_stack.push(obj.transformation.matrix)
            mvp_matrix = transform_stack.top()

            for triangle in obj.geometry:
                v0 = triangle.v0
                v1 = triangle.v1
                v2 = triangle.v2

                # Setting up arrays
                pos0 = [[v0.pos[0]], [v0.pos[1]], [v0.pos[2]], [1]]
                pos1 = [[v1.pos[0]], [v1.pos[1]], [v1.pos[2]], [1]]
                pos2 = [[v2.pos[0]], [v2.pos[1]], [v2.pos[2]], [1]]

                # Applying vertex and normal transformations.
                pos0 = np.matmul(mvp_matrix, pos0)
                pos1 = np.matmul(mvp_matrix, pos1)
                pos2 = np.matmul(mvp_matrix, pos2)

                # Conversion to row vectors and normalization.
                pos0 = np.transpose(pos0)[0]
                pos1 = np.transpose(pos1)[0]
                pos2 = np.transpose(pos2)[0]

                # Homogenizing all the vectors.
                pos0 = pos0 / pos0[3]
                pos1 = pos1 / pos1[3]
                pos2 = pos2 / pos2[3]

                # Converting vertices to raster space.
                pos0 = np.array([(pos0[0] + 1) * ((light.camera.resolution[0] - 1) / 2),
                                (pos0[1] + 1) * ((light.camera.resolution[1] - 1) / 2),
                                pos0[2]])
                pos1 = np.array([(pos1[0] + 1) * ((light.camera.resolution[0] - 1) / 2),
                                (pos1[1] + 1) * ((light.camera.resolution[1] - 1) / 2),
                                pos1[2]])
                pos2 = np.array([(pos2[0] + 1) * ((light.camera.resolution[0] - 1) / 2),
                                (pos2[1] + 1) * ((light.camera.resolution[1] - 1) / 2),
                                pos2[2]])

                # Getting the co-ordinates from vectors.
                x0, y0, z0 = pos0
                x1, y1, z1 = pos1
                x2, y2, z2 = pos2

                # Getting the bounding box of the triangle.
                xmin = max(math.floor(min(x0, x1, x2)), 0)
                xmax = min(math.ceil(max(x0, x1, x2)), light.camera.resolution[0])
                ymin = max(math.floor(min(y0, y1, y2)), 0)
                ymax = min(math.ceil(max(y0, y1, y2)), light.camera.resolution[1])

                # Skip scan conversion if the area covered by the triangle is 0.
                if triangle_area((x0, y0), (x1, y1), (x2, y2)) == 0:
                    return

                # Scan converting the triangle.
                for y in range(ymin, ymax):
                    for x in range(xmin, xmax):
                        # Getting the barycentric co-ordinates of the pixel.
                        alpha = triangle_area((x, y), (x1, y1), (x2, y2)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
                        beta = triangle_area((x, y), (x2, y2), (x0, y0)) / triangle_area((x0, y0), (x1, y1), (x2, y2))
                        gamma = triangle_area((x, y), (x0, y0), (x1, y1)) / triangle_area((x0, y0), (x1, y1), (x2, y2))

                        # Interpolating the Z value.
                        z = alpha * z0 + beta * z1 + gamma * z2

                        if alpha >= 0 and beta >= 0 and gamma >= 0:
                            # Updating the shadow buffer value.
                            light.set_shadow_buffer(x, y, z)

            # Popping the object transformation off the stack.
            transform_stack.pop()


def get_fragment_shadow(w_pos, w_normal, lights, bias=0.005):
    """
    Method to get the amount of shadow cast on a fragment by all lights in the scene.

    Args:
        w_pos(list): World space position of the fragment.
        w_normal(list): World space normal of the fragment.
        lights(list): List of lights in the scene.
        bias(float): Bias to use when checking whether a fragment is in shadow. Defaults to 0.005.

    Returns:
        (float): Shadow value of the fragment.

    """
    shadow = 0.0
    num_directional_lights = 0

    for light in lights:
        if light.type == "ambient":
            continue

        num_directional_lights += 1

        # Getting the position in light projection space.
        light_transform_stack = TransformStack()
        light_transform_stack.push(light.camera.projection_matrix)
        light_transform_stack.push(light.camera.cam_matrix)
        l_pos = np.matmul(light_transform_stack.top(), w_pos)

        # Calculating the shadow bias.
        lv_pos = np.matmul(light.camera.cam_matrix, w_pos)
        l_norm = np.matmul(light.camera.cam_matrix, w_normal)
        l_norm = l_norm / np.sqrt(np.dot(l_norm, l_norm))
        l_dir = -lv_pos / np.sqrt(np.dot(lv_pos, lv_pos))
        shadow_bias = max(0.05 * (1.0 - np.dot(l_norm, l_dir)), bias)

        # Homogenizing all the position vector.
        l_pos = l_pos / l_pos[3]

        # Conversion to light raster space.
        x = round((l_pos[0] + 1) * ((light.camera.resolution[0] - 1) / 2))
        y = round((l_pos[1] + 1) * ((light.camera.resolution[1] - 1) / 2))
        z = l_pos[2]

        if (x < 0) or (y < 0) or (x > light.camera.resolution[0] - 1) or (y > light.camera.resolution[1] - 1):
            continue

        # Accumulate the shadow value if the fragment is occluded in the light shadow buffer.
        z_shadow = light.get_shadow_buffer_depth(x, y)
        if z - shadow_bias > z_shadow:
            shadow += 1.0

    if num_directional_lights == 0:
        return shadow

    return shadow / num_directional_lights


def wireframe_shader(image, camera, pos0, pos1, pos2, color=None):
    """
    Method implementing a wireframe shader.

    Args:
        image(Image): Image to write the pixels into.
        camera(Camera): Camera that is viewing the scene.
        pos0(list): Raster space position of vertex 0.
        pos1(list): Raster space position of vertex 1.
        pos2(list): Raster space position of vertex 2.
        color(list): Color to use for drawing the wireframe lines. Defaults to None.

    """
    # Getting the co-ordinates from vectors.
    x0, y0, z0 = pos0
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    line_color = color if color is not None else (0, 0, 0)

    steep = False
    # Swap x and y for slope > 45.
    if abs(x1 - x2) < abs(y1 - y2) :
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        steep = True

    # Step from smaller x to bigger x.
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = y2 - y1

    derr = abs(2 * dy)
    derr2 = 0

    x = math.floor(x1)
    y = math.floor(y1)

    # Bressenham's algorithm for line rendering.
    while x <= math.floor(x2) and x < camera.resolution[0] - 1 and y < camera.resolution[1] - 1:
        if (steep):
            image.putpixel((y, -x), line_color)
        else:
            image.putpixel((x, -y), line_color)

        derr2 += derr
        if derr2 > dx:
            y += 1 if y2 > y1 else -1
            derr2 -= dx * 2
        x += 1
