import math
import os
import numpy as np

from PIL import Image, ImageOps

alpha, beta, gamma = 0,0,0

# Defining line equation functions
def f01(xfun, yfun, x1, x0, y1, y0):
        return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0


def f12(xfun, yfun, x1, x2, y1, y2):
        return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1


def f20(xfun, yfun, x0, x2, y0, y2):
        return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2

def halftoneShading(x,y,camera,material, vert1, vert2, vert3, lights, N):
    rgb=[0,0,0]
    normal =N
    lightVec = [0,0,0]
    diffSum = [0,0,0]
    amb = [0,0,0]
    halfVec=[0,0,0]
    specSum=[0,0,0]

    col=material["Cs"]

    darkCol = [0,0,0]
    lightCol = [0,0,0]
    for i in range(3):
        darkCol[i]= col[i]*0.15*255
        lightCol[i]= col[i]*0.7*255
    

    #define and normalize E vector for lighting
    global eVec
    eVec = [0,0,0]
    cf = np.array(camera["from"])
    ct = np.array(camera["to"])
    for i in range(3):
        eVec[i] = (cf[i] - ct[i]).item()
    eVec = eVec/np.linalg.norm(eVec)

    # Getting and transposing normal
    normal = np.transpose(N[:-1])[0]
    normal = normal / np.linalg.norm(normal)
        
    for light in lights: 
        light["color"] = [1,1,1]
        if(light["type"] == "ambient"):
            #ambient component
            for i in range(3):
                amb[i] = light["color"][i]*material["Ka"]*light["intensity"]
        elif(light["type"] == "directional"): 
            #creating light vector
            lf = np.array(light["from"])
            lt = np.array(light["to"])


            for i in range(3):
                lightVec[i] = (lf[i]-lt[i]).item()
            lightVec = lightVec / np.linalg.norm(lightVec)

            #NdotL quantization
            NdotE = 1 - np.dot(normal,eVec)
            NdotL = np.dot(normal,lightVec)

            for i in range(3):
                halfVec[i] = (lf[i]-lt[i]+cf[i]-ct[i]).item()
            NdotH = np.dot(normal,halfVec/np.linalg.norm(halfVec))

            for i in range(3):
                #diffuse component
                diffSum[i] += light["color"][i]*light["intensity"]*NdotL
                #specular component
                specSum[i] += light["color"][i]*light["intensity"]*((NdotH*NdotL)**material["n"])
    
    for i in range(3):
        rgb[i] = (amb[i] + (material["Kd"]*diffSum[i])) + (material["Ks"]*specSum[i]) #+ fresnel
        
    val = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]

    for i in range(3):
        rgb[i]*=255

    x/=height
    y/=height
    x *= 100.0
    y *= 100.0
    fracX = (x%1) - 0.5
    fracY = (y%1) - 0.5
    
    len = math.sqrt((fracX**2) + (fracY**2))
    
    val = 0.5 if val>0.5 else val
    
    if(not (len < math.sqrt(0.5 - val))):
        rgb= [255,255,255]
    else:
        rgb = darkCol if val <0.15 else  lightCol
            
    if(val <0.45):
        return rgb
    else:
        return [255,255,255]



def renderGeom(camera, triangle_data, material_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix, light_data):

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
                    # Checking pixel z depth
                    if z < zbuffer[x, y]:
                        #  Shading
                        cp_h = [0, 0, 0]
                        pixel_N = [0, 0, 0]

                        for i in range(3):
                            # Phong Shading - interpolating normal
                            pixel_N[i] = (alpha * n0_array[i] + beta * n1_array[i] + gamma * n2_array[i]).item()

                        cp_h = halftoneShading(x,y,camera, material_data, [x0,y0,z0], [x1,y1,z1],[x2,y2,z2], light_data, [[pixel_N[0]], [pixel_N[1]], [pixel_N[2]], [1]])

                        im.putpixel((x, -y),
                            (round(np.nan_to_num(cp_h[0])), round(np.nan_to_num(cp_h[1])),
                            round(np.nan_to_num(cp_h[2]))))
                        zbuffer[x, y] = z



def renderHalftone(camera, triangle_data, material_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix, light_data):

    print("===================\nHalftone Render Begin\n")
    global im, width, height, zbuffer, stencilBuffer
    im = Image.new('RGB', [xres, yres], 0x000000)
    width, height = im.size
    zbuffer = np.matrix(np.ones((xres, yres)) * np.inf)

    renderGeom(camera, triangle_data, material_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix, light_data)

    im.show()
    print("Halftone Render Successful\n===================\n")