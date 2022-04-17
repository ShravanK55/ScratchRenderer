"""
Module implementing shader functions.
"""

import math
import numpy as np
from utils import triangle_area


def light_calc(obj, camera, lights, n):
    """
    Method to calculate the color of a fragment from lighting.

    Args:
        obj(Object): Object being rendered.
        camera(Camera): Camera that is viewing the object.
        lights(list): List of lights in the scene.
        n(list): Normal vector of the fragment being shaded.

    Returns:
        (list): Color of the fragment.

    """
    ambient_color = np.array([0, 0, 0])
    diffuse_color = np.array([0, 0, 0])
    specular_color = np.array([0, 0, 0])

    for light in lights:
        light_color = np.array(light.color) * light.intensity

        if light.type == "ambient":
            ambient_color = light_color
            continue

        # Getting the light, normal, eye direction and reflected light vectors.
        l = light.direction
        nt = np.transpose(n[:-1])[0]
        e = camera.direction
        r = 2 * np.dot(nt, l) * nt - l
        r = r / np.sqrt(np.dot(r, r))

        # Calculating the N dot L, N dot E and R dot E products.
        n_dot_l = np.dot(nt, l)
        n_dot_e = np.dot(nt, e)
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
    color = (obj.ka * ambient_color) + (obj.kd * diffuse_color) + (obj.ks * specular_color)
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
    uv0 = uv0 / pos0[2]
    uv1 = uv1 / pos1[2]
    uv2 = uv2 / pos2[2]

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
                u = alpha * uv0[0] + beta * uv1[0] + gamma * uv2[0]
                v = alpha * uv0[1] + beta * uv1[1] + gamma * uv2[1]
                z_at_pixel = 1 / (alpha * (1 / z0) + beta * (1 / z1) + gamma * (1 / z2))
                u = u * z_at_pixel
                v = v * z_at_pixel

                """
                TODO: Implement texture manager.
                # Bilinear interpolation to get texture RGB
                x_location = min(u * (tex_xres - 1), tex_xres - 2)
                y_location = min(v * (tex_yres - 1), tex_yres - 2)
                x_location = max(x_location, 0)
                y_location = max(y_location, 0)
                """

                # Shading the pixel.
                if z < z_buffer[x, y]:
                    n = alpha * n0 + beta * n1 + gamma * n2
                    color = light_calc(obj, camera, lights, n)
                    image.putpixel((x, -y),
                                   (round(color[0]),
                                    round(color[1]),
                                    round(color[2])))
                    z_buffer[x, y] = z
