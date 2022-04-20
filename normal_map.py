import math
import os
import numpy as np
from PIL import Image, ImageOps

# Light calculation function
def light_calc(N, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s):
    L = light_n
    ambient_light = np.array(list(la)) * la_intensity
    directional_light = np.array(list(ld)) * ld_intensity

    # Getting and transposing normal
    N = np.transpose(N[:-1])[0]
    N = N / np.linalg.norm(N)

    # Calculating N dot L and N dot E
    NdotL = np.dot(N, L)
    NdotE = np.dot(N, E)

    if NdotL >= 0 and NdotE >= 0:
        N = N
    elif NdotL < 0 and NdotE < 0:
        N = [-normal for normal in N]
        N = np.array(N)

    # Calculating and normalizing R
    R = 2 * np.dot(N, L) * N - L
    R = R / np.linalg.norm(R)

    # Clamping R dot E
    RdotE = max(np.dot(R, E), 0)
    RdotE = min(RdotE, 1)

    # Calculating specular, diffuse, and ambient
    specular = Ks * directional_light * (RdotE ** s)
    diffuse = (Kd * directional_light * np.dot(N, L))
    ambient = (Ka * ambient_light)

    if (NdotL < 0 and NdotE >= 0) or (NdotL >= 0 and NdotE < 0):
        diffuse = 0

    color = specular + ( (diffuse + ambient))

    return color


# Defining line equation functions
def f01(xfun, yfun, x1, x0, y1, y0):
    return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0


def f12(xfun, yfun, x1, x2, y1, y2):
    return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1


def f20(xfun, yfun, x0, x2, y0, y2):
    return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2


# Generate a depth map
def generate_depth_map(triangle_data, plane_data, mvp_matrix, xres, yres):
    depth_map_im = Image.new('RGB', [xres, yres], 0x000000)
    depth_map = [[0 for x in range(xres)] for y in range(yres)]

    z_buffer = np.matrix(np.ones((xres, yres)) * np.inf)

    # Triangle normals
    for triangle in triangle_data.get('data'):
        # Getting x,y,z values from each vertex
        x0 = triangle.get('v0').get('v')[0]
        y0 = triangle.get('v0').get('v')[1]
        x1 = triangle.get('v1').get('v')[0]
        y1 = triangle.get('v1').get('v')[1]
        x2 = triangle.get('v2').get('v')[0]
        y2 = triangle.get('v2').get('v')[1]
        z0 = triangle.get('v0').get('v')[2]
        z1 = triangle.get('v1').get('v')[2]
        z2 = triangle.get('v2').get('v')[2]

        # Setting up arrays
        v0_array = [[x0], [y0], [z0], [1]]
        v1_array = [[x1], [y1], [z1], [1]]
        v2_array = [[x2], [y2], [z2], [1]]

        # Applying transformations 0 -> NDC
        # Vertex transformations
        v0_array = np.matmul(mvp_matrix, v0_array)
        v1_array = np.matmul(mvp_matrix, v1_array)
        v2_array = np.matmul(mvp_matrix, v2_array)

        # Divide by w
        x0 = v0_array[0] / v0_array[3]
        y0 = v0_array[1] / v0_array[3]
        x1 = v1_array[0] / v1_array[3]
        y1 = v1_array[1] / v1_array[3]
        x2 = v2_array[0] / v2_array[3]
        y2 = v2_array[1] / v2_array[3]
        z0 = v0_array[2] / v0_array[3]
        z1 = v1_array[2] / v1_array[3]
        z2 = v2_array[2] / v2_array[3]

        #  NDC -> Raster
        x0 = ((x0 + 1) * ((xres - 1) / 2))
        y0 = ((y0 + 1) * ((yres - 1) / 2))
        x1 = ((x1 + 1) * ((xres - 1) / 2))
        y1 = ((y1 + 1) * ((yres - 1) / 2))
        x2 = ((x2 + 1) * ((xres - 1) / 2))
        y2 = ((y2 + 1) * ((yres - 1) / 2))

        # Setting mins and maxs while checking for bounds
        xmin = max(math.floor(min(x0, x1, x2)), 0)
        xmax = min(math.ceil(max(x0, x1, x2)), xres)
        ymin = max(math.floor(min(y0, y1, y2)), 0)
        ymax = min(math.ceil(max(y0, y1, y2)), yres)

        if f12(x0, y0, x1, x2, y1, y2) == 0 or f20(x1, y1, x0, x2, y0, y2) == 0 or f01(x2, y2, x1, x0, y1, y0) == 0:
            continue

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                # Calculating barycentric coordinates
                alpha = f12(x, y, x1, x2, y1, y2) / f12(x0, y0, x1, x2, y1, y2)
                beta = f20(x, y, x0, x2, y0, y2) / f20(x1, y1, x0, x2, y0, y2)
                gamma = f01(x, y, x1, x0, y1, y0) / f01(x2, y2, x1, x0, y1, y0)

                # Calculating z value
                z = alpha * z0 + beta * z1 + gamma * z2

                # Filling pixel based on barycentric coordinates
                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    # Checking pixel z depth
                    if z < z_buffer[x, y]:
                        # Update z-buffer
                        z_buffer[x, y] = z

                        # Map the depth to [0, 255]
                        depth_mapped = (z % 1) * 255
                        depth_map[x][-y] = depth_mapped

                        # Write this depth as a pixel to a PNG image
                        depth_map_im.putpixel((x, -y), (int(depth_mapped), int(depth_mapped), int(depth_mapped)))

    # Show the depth map image
    depth_map_im.show()

    # Return the depth map
    return depth_map


# Generate a normal map from a depth map
def generate_normal_map(depth_map, xres, yres):
    normal_map_im = Image.new('RGB', [xres, yres], 0x000000)
    normal_map = [[0] * xres] * yres

    # Calculate the normal based on the depth on each point
    for x in range(0, xres - 1):
        for y in range(0, yres - 1):
            # Get the depth on the pixel
            depth = depth_map[x][y]

            # Calculate the vector perpendicular to the X plane
            differentialX = (depth_map[x + 1][y] - depth_map[x - 1][y]) / 2.0
            tangentX = [1, 0, differentialX]

            # Calculate the vector perpendicular to the Y plane
            differentialY = (depth_map[x][y + 1] - depth_map[x][y - 1]) / 2.0
            tangentY = [0, 1, differentialY]

            # Calculate the normal vector
            normal = [tangentX[1] * tangentY[2] - tangentX[2] * tangentY[1],
                      tangentX[2] * tangentY[0] - tangentX[0] * tangentY[2],
                      tangentX[0] * tangentY[1] - tangentX[1] * tangentY[0]]

            # Normalize the normal vector
            normal_length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
            normal[0] = normal[0] / normal_length
            normal[1] = normal[1] / normal_length
            normal[2] = normal[2] / normal_length

            # Map the normal from [-1, 1] to [0, 1]
            normalX_mapped = normal[0] * 0.5 + 0.5
            normalY_mapped = normal[1] * 0.5 + 0.5
            normalZ_mapped = normal[2] * 0.5 + 0.5

            # Convert the normal to a pixel[0, 255] on a PNG image
            normalX_mapped = normalX_mapped * 255
            normalY_mapped = normalY_mapped * 255
            normalZ_mapped = normalZ_mapped * 255
            normal_map[x][y] = (normalX_mapped, normalY_mapped, normalZ_mapped)

            # Write this normal as a pixel to a PNG image
            normal_map_im.putpixel((x, y), (int(normalX_mapped), int(normalY_mapped), int(normalZ_mapped)))

    # Show the normal map image
    normal_map_im.show(normal_map_im)

    # Return the normal map
    return normal_map_im


# Render the scene using the normal map
def render_with_normal_map(normal_map_im, triangle_data, mvp_matrix, normal_transformation_matrix, xres, yres, texmap,
                           tex_xres, tex_yres, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s):
    im = Image.new('RGB', [xres, yres], 0x000000)

    zbuffer = np.matrix(np.ones((xres, yres)) * np.inf)

    # Getting triangle data
    for triangle in triangle_data.get('data'):

        # Getting x,y,z values from each vertex
        x0 = triangle.get('v0').get('v')[0]
        y0 = triangle.get('v0').get('v')[1]
        x1 = triangle.get('v1').get('v')[0]
        y1 = triangle.get('v1').get('v')[1]
        x2 = triangle.get('v2').get('v')[0]
        y2 = triangle.get('v2').get('v')[1]
        z0 = triangle.get('v0').get('v')[2]
        z1 = triangle.get('v1').get('v')[2]
        z2 = triangle.get('v2').get('v')[2]

        # Getting normals
        nx0 = triangle.get('v0').get('n')[0]
        ny0 = triangle.get('v0').get('n')[1]
        nz0 = triangle.get('v0').get('n')[2]
        nx1 = triangle.get('v1').get('n')[0]
        ny1 = triangle.get('v1').get('n')[1]
        nz1 = triangle.get('v1').get('n')[2]
        nx2 = triangle.get('v2').get('n')[0]
        ny2 = triangle.get('v2').get('n')[1]
        nz2 = triangle.get('v2').get('n')[2]

        # Getting texture coords
        u0 = triangle.get('v0').get('t')[0]
        v0 = triangle.get('v0').get('t')[1]
        u1 = triangle.get('v1').get('t')[0]
        v1 = triangle.get('v1').get('t')[1]
        u2 = triangle.get('v2').get('t')[0]
        v2 = triangle.get('v2').get('t')[1]

        # Setting up arrays
        v0_array = [[x0], [y0], [z0], [1]]
        v1_array = [[x1], [y1], [z1], [1]]
        v2_array = [[x2], [y2], [z2], [1]]
        n0_array = [[nx0], [ny0], [nz0], [1]]
        n1_array = [[nx1], [ny1], [nz1], [1]]
        n2_array = [[nx2], [ny2], [nz2], [1]]

        # Applying transformations 0 -> NDC
        # Vertex transformations
        v0_array = np.matmul(mvp_matrix, v0_array)
        v1_array = np.matmul(mvp_matrix, v1_array)
        v2_array = np.matmul(mvp_matrix, v2_array)

        n0_array = np.matmul(normal_transformation_matrix, n0_array)
        n1_array = np.matmul(normal_transformation_matrix, n1_array)
        n2_array = np.matmul(normal_transformation_matrix, n2_array)

        # Divide by w
        x0 = v0_array[0] / v0_array[3]
        y0 = v0_array[1] / v0_array[3]
        x1 = v1_array[0] / v1_array[3]
        y1 = v1_array[1] / v1_array[3]
        x2 = v2_array[0] / v2_array[3]
        y2 = v2_array[1] / v2_array[3]
        z0 = v0_array[2] / v0_array[3]
        z1 = v1_array[2] / v1_array[3]
        z2 = v2_array[2] / v2_array[3]

        # Getting normals in world space
        nx0 = n0_array[0][0]
        ny0 = n0_array[1][0]
        nz0 = n0_array[2][0]
        nx1 = n1_array[0][0]
        ny1 = n1_array[1][0]
        nz1 = n1_array[2][0]
        nx2 = n2_array[0][0]
        ny2 = n2_array[1][0]
        nz2 = n2_array[2][0]

        # For Gouraud - calculating c1,c2,c3 - one for each vertex
        c1 = light_calc(n0_array, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s)
        c2 = light_calc(n1_array, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s)
        c3 = light_calc(n2_array, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s)

        #  NDC -> Raster
        x0 = ((x0 + 1) * ((xres - 1) / 2))
        y0 = ((y0 + 1) * ((yres - 1) / 2))
        x1 = ((x1 + 1) * ((xres - 1) / 2))
        y1 = ((y1 + 1) * ((yres - 1) / 2))
        x2 = ((x2 + 1) * ((xres - 1) / 2))
        y2 = ((y2 + 1) * ((yres - 1) / 2))

        # For Perspective correction, dividing by its own z
        u0 = u0 / z0
        u1 = u1 / z1
        u2 = u2 / z2

        # z-buffer and scan-converter

        # Setting mins and maxs while checking for bounds
        xmin = max(math.floor(min(x0, x1, x2)), 0)
        xmax = min(math.ceil(max(x0, x1, x2)), xres)
        ymin = max(math.floor(min(y0, y1, y2)), 0)
        ymax = min(math.ceil(max(y0, y1, y2)), yres)

        if f12(x0, y0, x1, x2, y1, y2) == 0 or f20(x1, y1, x0, x2, y0, y2) == 0 or f01(x2, y2, x1, x0, y1, y0) == 0:
            continue

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                # Calculating barycentric coordinates
                alpha = f12(x, y, x1, x2, y1, y2) / f12(x0, y0, x1, x2, y1, y2)
                beta = f20(x, y, x0, x2, y0, y2) / f20(x1, y1, x0, x2, y0, y2)
                gamma = f01(x, y, x1, x0, y1, y0) / f01(x2, y2, x1, x0, y1, y0)

                # Calculating z value
                z = alpha * z0 + beta * z1 + gamma * z2

                # Filling pixel based on barycentric coordinates
                if alpha >= 0 and beta >= 0 and gamma >= 0:

                    # Barycentric interpolation
                    u = alpha * u0 + beta * u1 + gamma * u2
                    v = alpha * v0 + beta * v1 + gamma * v2

                    # Calculating z at pixel
                    z_at_pixel = 1 / (alpha * (1 / z0) + beta * (1 / z1) + gamma * (1 / z2))

                    # Multiplying u,v by z_at_pixel to get usable u,v
                    u = u * z_at_pixel
                    v = v * z_at_pixel

                    # Bilinear interpolation to get texture RGB
                    x_location = min(u * (tex_xres - 1), tex_xres - 2)
                    y_location = min(v * (tex_yres - 1), tex_yres - 2)
                    x_location = max(x_location, 0)
                    y_location = max(y_location, 0)

                    if isinstance(x_location, np.ndarray):
                        x_location.astype(int)
                        x_location = x_location[0]

                    if isinstance(y_location, np.ndarray):
                        y_location.astype(int)
                        y_location = y_location[0]

                    f = x_location - np.trunc(x_location)
                    g = y_location - np.trunc(y_location)

                    p00RGB = np.array(texmap.getpixel((math.trunc(x_location), math.trunc(y_location))))
                    p11RGB = np.array(texmap.getpixel((math.trunc(x_location) + 1, math.trunc(y_location) + 1)))
                    p10RGB = np.array(texmap.getpixel((math.trunc(x_location) + 1, math.trunc(y_location))))
                    p01RGB = np.array(texmap.getpixel((math.trunc(x_location) + 1, math.trunc(y_location))))

                    p0010RGB = f * p10RGB + (1 - f) * p00RGB
                    p0111RGB = f * p11RGB + (1 - f) * p01RGB
                    pOutputRGB = g * p0111RGB + (1 - g) * p0010RGB

                    Kt = 0.7

                    # Checking pixel z depth
                    if z < zbuffer[x, y]:

                        #  Shading
                        cp_g = [0, 0, 0]
                        pixel_N = [0, 0, 0]
                        for i in range(0, len(c1)):
                            # Phong Shading - interpolating normal
                            pixel_N[i] = (alpha * n0_array[i] + beta * n1_array[i] + gamma * n2_array[i]).item()

                        # Normal map
                        # Unbias the normal
                        normal_map_normal = normal_map_im.getpixel((x, -y))
                        normal_map_normalX = (normal_map_normal[0] / 255) * 2.0 - 1.0
                        normal_map_normalY = (normal_map_normal[1] / 255) * 2.0 - 1.0
                        normal_map_normalZ = (normal_map_normal[2] / 255) * 2.0 - 1.0

                        # Phong Shading
                        cp_ph = light_calc([[normal_map_normalX], [normal_map_normalY], [normal_map_normalZ], [0]], light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s)

                        # Phong
                        im.putpixel((x, -y),
                                    (round((cp_ph[0]) * pOutputRGB[0] * Kt),
                                     round((cp_ph[1]) * pOutputRGB[1] * Kt),
                                     round((cp_ph[2]) * pOutputRGB[2] * Kt)))
                        zbuffer[x, y] = z
    im.show()
