import math
import numpy as np


def createShadowBuffer(triangle_data, xres, yres, light_mvp_matrix, shadow_buffer):
    for triangle in triangle_data:
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

        # Applying transformations 0 -> W
        # Vertex transformations

        v0_array = np.matmul(light_mvp_matrix, v0_array)
        v1_array = np.matmul(light_mvp_matrix, v1_array)
        v2_array = np.matmul(light_mvp_matrix, v2_array)

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

        # z-buffer and scan-converter
        # Defining line equation functions
        def f01(xfun, yfun):
            return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0

        def f12(xfun, yfun):
            return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1

        def f20(xfun, yfun):
            return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2

        # Setting mins and maxs while checking for bounds
        xmin = max(math.floor(min(x0, x1, x2)), 0)
        xmax = min(math.ceil(max(x0, x1, x2)), xres)
        ymin = max(math.floor(min(y0, y1, y2)), 0)
        ymax = min(math.ceil(max(y0, y1, y2)), yres)

        if f12(x0, y0) == 0 or f20(x1, y1) == 0 or f01(x2, y2) == 0:
            continue

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                # Calculating barycentric coordinates

                alpha = f12(x, y) / f12(x0, y0)
                beta = f20(x, y) / f20(x1, y1)
                gamma = f01(x, y) / f01(x2, y2)

                # Calculating z value
                z = alpha * z0 + beta * z1 + gamma * z2

                # Filling pixel based on barycentric coordinates
                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    # Checking pixel "shadow depth"
                    if z < shadow_buffer[x, y]:
                        shadow_buffer[x, y] = z


def renderShadow(im, triangle_data, xres, yres, mvp_matrix, light_mvp_matrix, shadow_buffer, zbuffer, la, la_intensity, Ka):
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

        # W -> NDC

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

        # z-buffer and scan-converter
        # Defining line equation functions
        def f01(xfun, yfun):
            return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0

        def f12(xfun, yfun):
            return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1

        def f20(xfun, yfun):
            return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2

        # Setting mins and maxs while checking for bounds
        xmin = max(math.floor(min(x0, x1, x2)), 0)
        xmax = min(math.ceil(max(x0, x1, x2)), xres)
        ymin = max(math.floor(min(y0, y1, y2)), 0)
        ymax = min(math.ceil(max(y0, y1, y2)), yres)

        if f12(x0, y0) == 0 or f20(x1, y1) == 0 or f01(x2, y2) == 0:
            continue

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):

                # Calculating barycentric coordinates

                alpha = f12(x, y) / f12(x0, y0)
                beta = f20(x, y) / f20(x1, y1)
                gamma = f01(x, y) / f01(x2, y2)

                # Calculating z value
                z = alpha * z0 + beta * z1 + gamma * z2

                # Converting pixel to light space
                # Raster -> NDC
                NDC_x = (x / ((xres - 1) / 2)) - 1
                NDC_y = (y / ((yres - 1) / 2)) - 1

                # NDC -> C -> W
                v = np.matmul(np.linalg.inv(mvp_matrix), [NDC_x, NDC_y, z, 1])

                # W -> L -> NDC
                v = np.matmul(light_mvp_matrix, v)
                x_light = ((v[0] / v[3])[0])
                y_light = ((v[1] / v[3])[0])
                z_light = v[2] / v[3]

                # NDC - > Raster
                x_light = round((x_light + 1) * ((xres - 1) / 2))
                y_light = round((y_light + 1) * ((yres - 1) / 2))

                # Filling pixel based on barycentric coordinates
                if alpha >= 0 and beta >= 0 and gamma >= 0:
                    # If out of bounds in shadow map then that pixel is in light
                    if x_light > xres - 1 or y_light > yres - 1 or x_light < 0 or y_light < 0:
                        continue
                    # Checking pixel's "shadow depth"
                    if z_light[0] > shadow_buffer[x_light, y_light]:
                        # Checking pixel's z-depth
                        if z < zbuffer[x, y]:
                            ambient_light = np.array(list(la)) * la_intensity * Ka
                            rgbPixel = ([112, 86, 15] * ambient_light)
                            im.putpixel((x, -y),(round(rgbPixel[0]) ,round(rgbPixel[1]) ,round((rgbPixel[2]))))
                            zbuffer[x, y] = z


