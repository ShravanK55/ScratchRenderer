import json
import math

import numpy as np
from PIL import Image


def renderPlane(im, xres, yres, zbuffer,camera_matrix, perspective_matrix):
    with open("plane.json") as json_file:
        triangle_data = json.load(json_file)

    '''im = Image.new('RGB', [xres, yres], 0x000000)
    width, height = im.size
    for y in range(height):
        for x in range(width):
            im.putpixel((x, y), (128, 128, 128))
    '''

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

        # Setting up arrays
        v0_array = [[x0], [y0], [z0], [1]]
        v1_array = [[x1], [y1], [z1], [1]]
        v2_array = [[x2], [y2], [z2], [1]]
        n0_array = [[nx0], [ny0], [nz0], [1]]
        n1_array = [[nx1], [ny1], [nz1], [1]]
        n2_array = [[nx2], [ny2], [nz2], [1]]

        # W -> C

       # print(v0_array)
        #print(v1_array)
        #print(v2_array)
        v0_array = np.matmul(camera_matrix, v0_array)
        v1_array = np.matmul(camera_matrix, v1_array)
        v2_array = np.matmul(camera_matrix, v2_array)


        # C -> NDC

        v0_array = np.matmul(perspective_matrix, v0_array)
        v1_array = np.matmul(perspective_matrix, v1_array)
        v2_array = np.matmul(perspective_matrix, v2_array)

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

        # Getting normals
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
        # c1 = light_calc(n0_array)
        # c2 = light_calc(n1_array)
        # c3 = light_calc(n2_array)

        #  NDC -> Raster
        x0 = ((x0 + 1) * ((xres - 1) / 2))
        y0 = ((y0 + 1) * ((yres - 1) / 2))
        x1 = ((x1 + 1) * ((xres - 1) / 2))
        y1 = ((y1 + 1) * ((yres - 1) / 2))
        x2 = ((x2 + 1) * ((xres - 1) / 2))
        y2 = ((y2 + 1) * ((yres - 1) / 2))
        #print('light coords in plane renderer')
        #print(x0, x1, x2)
        #print(y0, y1, y2)
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

                    # Checking pixel z depth
                    if z < zbuffer[x, y]:
                        im.putpixel((x, -y),
                                    (112, 86, 15))




