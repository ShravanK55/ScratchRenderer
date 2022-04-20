import math
import os
import numpy as np

from PIL import Image, ImageOps


def draw_line(x1, y1, x2, y2):
    black = 0,0,0

    steep = False
    # if slope is more than 45, step along y. Swap x and y
    if abs(x1-x2)<abs(y1-y2) : 
        x1, y1 = y1, x1 
        x2, y2 = y2, x2 
        steep = True 
    
    #step from smaller x to bigger x
    if x1>x2:
        x1, x2 = x2, x1 
        y1, y2 = y2, y1 

    
    dx = x2-x1
    dy = y2-y1
    
    derr = abs(2*dy)
    derr2 = 0
    
    x=math.floor(x1)
    y=math.floor(y1)
    
    #Bressenham's Algorithm for Line Rendering
    while x <= math.floor(x2) and x<width-1 and y<height-1:
        if (steep): 
            im.putpixel( (y,-x),black)
        else:
            im.putpixel( (x,-y),black) 
 
        derr2 += derr; 
        if derr2 > dx : 
            y += 1 if y2>y1 else -1
            derr2 -= dx*2
        x+=1

def renderGeom(triangle_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix):

    #set background color
    for x in range(width):
        for y in range(height):
            im.putpixel( (x,y),(255, 255, 255))

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

        # Applying transformations 0 -> W

        # Vertex transformations

        v0_array = np.matmul(rotation_matrix, v0_array)
        v1_array = np.matmul(rotation_matrix, v1_array)
        v2_array = np.matmul(rotation_matrix, v2_array)

        v0_array = np.matmul(scale_matrix, v0_array)
        v1_array = np.matmul(scale_matrix, v1_array)
        v2_array = np.matmul(scale_matrix, v2_array)

        v0_array = np.matmul(translate_matrix, v0_array)
        v1_array = np.matmul(translate_matrix, v1_array)
        v2_array = np.matmul(translate_matrix, v2_array)

        # Normal transformations

        n0_array = np.matmul(scale_matrix_inverse_transpose, n0_array)
        n1_array = np.matmul(scale_matrix_inverse_transpose, n1_array)
        n2_array = np.matmul(scale_matrix_inverse_transpose, n2_array)

        n0_array = np.matmul(rotation_matrix, n0_array)
        n1_array = np.matmul(rotation_matrix, n1_array)
        n2_array = np.matmul(rotation_matrix, n2_array)

        # W -> C

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


        #  NDC -> Raster
        x0 = ((x0 + 1) * ((xres - 1) / 2))
        y0 = ((y0 + 1) * ((yres - 1) / 2))
        x1 = ((x1 + 1) * ((xres - 1) / 2))
        y1 = ((y1 + 1) * ((yres - 1) / 2))
        x2 = ((x2 + 1) * ((xres - 1) / 2))
        y2 = ((y2 + 1) * ((yres - 1) / 2))

        #Wireframe render
        draw_line(x0, y0, x1, y1)
        draw_line(x1, y1, x2, y2)
        draw_line(x2, y2, x0, y0)


def renderWireframe(triangle_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix):
    print("===================\nWireframe Render Begin\n")

    global im, width, height, zbuffer, stencilBuffer
    im = Image.new('RGB', [xres, yres], 0x000000)
    width, height = im.size
    zbuffer = np.matrix(np.ones((xres, yres)) * np.inf)

    renderGeom(triangle_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix) 

    im.show()
    print("Wireframe Render Successful\n===================\n")