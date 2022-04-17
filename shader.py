"""
Module implementing shader functions.
"""

import math
import numpy as np
from texture import get_texture_color
from utils import triangle_area


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


def geometry_pass_shader(image, g_buffer, obj, camera, lights, pos0, pos1, pos2, wpos0, wpos1, wpos2, n0, n1, n2, uv0,
                         uv1, uv2):
    """
    Method implementing a geometry pass shader (First pass in deferred rendering).

    Args:
        image(Image): Image to write the pixels into.
        g_buffer(GeometryBuffer): Geometry buffer of the image being rendered.
        obj(Object): Object being rendered.
        camera(Camera): Camera that is viewing the scene.
        lights(list): List of lights in the scene.
        pos0(list): Position of vertex 0.
        pos1(list): Position of vertex 1.
        pos2(list): Position of vertex 2.
        wpos0(list): World space position of vertex 0.
        wpos1(list): World space position of vertex 1.
        wpos2(list): World space position of vertex 2.
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
                # Barycentrically interpolating the world space position vectors.
                wpos = alpha * wpos0 + beta * wpos1 + gamma * wpos2

                # Getting the interpolated normal.
                n = alpha * n0 + beta * n1 + gamma * n2

                # Perspective correcting the UVs.
                uv = alpha * p_uv0 + beta * p_uv1 + gamma * p_uv2
                z_at_pixel = 1 / (alpha * (1 / z0) + beta * (1 / z1) + gamma * (1 / z2))
                uv = uv * z_at_pixel

                # Getting the object/texture color.
                color = obj.color
                if obj.texture:
                    color = get_texture_color(obj.texture, uv)

                # Interpolating the Z value.
                z = alpha * z0 + beta * z1 + gamma * z2

                # Adding geometry to the geometry buffer.
                success = g_buffer.set_attributes(x, y, wpos, n, color, z)

                # Shading the pixel.
                if success:
                    n = alpha * n0 + beta * n1 + gamma * n2
                    color = light_calc(obj, camera, lights, n, uv)
                    image.putpixel((x, -y), color)
