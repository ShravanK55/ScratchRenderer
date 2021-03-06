import math
import os
import numpy as np
from perlin_noise import PerlinNoise
import random

from PIL import Image, ImageOps

alpha, beta, gamma = 0,0,0

# Defining line equation functions
def f01(xfun, yfun, x1, x0, y1, y0):
        return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0


def f12(xfun, yfun, x1, x2, y1, y2):
        return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1


def f20(xfun, yfun, x0, x2, y0, y2):
        return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2

def toonShading(camera,material, vert1, vert2, vert3, lights, N):
    rgb=[0,0,0]
    #normal=[0,0,0]
    normal =N
    lightVec = [0,0,0]
    diffSum = [0,0,0]
    amb = [0,0,0]
    halfVec=[0,0,0]
    specSum=[0,0,0]
    #col=[1,0,0]
    col=material["Cs"]
    material["Ka"] = 0.5
    material["Kd"]=0.75
    material["Ks"]=0.9
    material["n"]=400.0

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
            light["intensity"]=0.2
            for i in range(3):
                amb[i] = light["color"][i]*material["Ka"]*light["intensity"]
        elif(light["type"] == "directional"):

            light["intensity"]=0.8
            #creating light vector
            lf = np.array(light["from"])
            lt = np.array(light["to"])


            for i in range(3):
                lightVec[i] = (lf[i]-lt[i]).item()
            lightVec = lightVec / np.linalg.norm(lightVec)

            #NdotL quantization
            NdotE = 1 - np.dot(normal,eVec)
            NdotL = np.dot(normal,lightVec)

            if NdotL <=0.01:
                NdotL = 0
            elif (NdotL >0.5 and NdotL<=1):
                NdotL=1
            else:
                NdotL=0.5

            fresnel = NdotE*NdotL
            fresnel = fresnel/((0.5*1.1)-(0.5))

            for i in range(3):
                halfVec[i] = (lf[i]-lt[i]+cf[i]-ct[i]).item()
            NdotH = np.dot(normal,halfVec/np.linalg.norm(halfVec))

            for i in range(3):
                #diffuse component
                diffSum[i] += light["color"][i]*light["intensity"]*NdotL
                #specular component
                specSum[i] += light["color"][i]*light["intensity"]*((NdotH*NdotL)**material["n"])

    for i in range(3):
        rgb[i] = (col[i]*(amb[i] + (material["Kd"]*diffSum[i]))) + (material["Ks"]*specSum[i]) #+ fresnel
    for i in range(3):
        rgb[i]*=255

    return rgb


def backface(camera, v0_array, v1_array, v2_array, rotation_matrix, scale_matrix, translate_matrix):
    v0=np.array(v0_array)
    v1=np.array(v1_array)
    v2=np.array(v2_array)
    
    normal = [0,0,0]
    vec1=[0,0,0]
    vec2=[0,0,0]
    viewVec=[0,0,0]

    camObj = [[camera["from"][0]],[camera["from"][1]],[camera["from"][2]],[1]]

    # Concatenated Object to World Transformation Matrix
    objToWorld_matrix = [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]
    objToWorld_matrix = np.matmul(rotation_matrix, objToWorld_matrix)
    objToWorld_matrix = np.matmul(scale_matrix, objToWorld_matrix)
    objToWorld_matrix = np.matmul(translate_matrix, objToWorld_matrix)

    #Inverse to get world to object Transformation Matrix
    worldToObj_matrix = np.linalg.inv(objToWorld_matrix)
    
    #Transform Camera from World to object space
    camObj = np.matmul(objToWorld_matrix, camObj)
    cam = np.array(camObj)
    #viewVec = matrixVecMul(worldToObj_matrix, camera["from"])
    
    for i in range(3):
        vec1[i] = (v1[i][0] - v0[i][0]).item()
        vec2[i] = (v2[i][0] - v0[i][0]).item()
        viewVec[i] = (v0_array[i][0] - cam[i][0]).item()
    

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)
    
    viewVec = viewVec / np.linalg.norm(viewVec)
    
    
    NdotE = np.dot(normal,viewVec)
    
    if(NdotE>(0.015)):
        return True
    
    return False


def renderGeom(camera, triangle_data, material_data, xres, yres,rotation_matrix, scale_matrix, translate_matrix, mvp_matrix, normal_transformation_matrix, light_data):
    
    #set background color
    for x in range(width):
        for y in range(height):
            imT.putpixel( (x,y),(255, 255, 255))
            imL.putpixel( (x,y),(255, 255, 255))

    #random starting point for perlin noise
    noi=random.uniform(0,1)

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


        #check if triangle is a backface
        backFace = backface(camera,v0_array,v1_array,v2_array, rotation_matrix, scale_matrix, translate_matrix,)

        #if traingle is a backFace, move the vertice in their normal direction as per the outline size
        if(backFace):
            #decide outline thickness using perlin noise
            outlineSize = -noise(noi)
            noi+=0.001 
            outlineSize = 0.04 + (0.1 - 0.04) * (outlineSize)

            #normalize normals
            n0_a = np.array(n0_array)
            n0_a = n0_a / np.linalg.norm(n0_a)

            n1_a = np.array(n1_array)
            n1_a = n1_a / np.linalg.norm(n1_a)

            n2_a = np.array(n2_array)
            n2_a = n2_a / np.linalg.norm(n2_a)

            #move vertices
            for i in range(3):
                v0_array[i][0] += n0_a[i][0] * outlineSize
                v1_array[i][0] += n1_a[i][0] * outlineSize
                v2_array[i][0] += n2_a[i][0] * outlineSize

        # Applying transformations 0 -> NDC
        # Vertex transformations
        v0_array = np.matmul(mvp_matrix, v0_array)
        v1_array = np.matmul(mvp_matrix, v1_array)
        v2_array = np.matmul(mvp_matrix, v2_array)

        # Normal transformations

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
                        cp_t = [0, 0, 0]
                        pixel_N = [0, 0, 0]

                        for i in range(3):
                            # Phong Shading - interpolating normal
                            pixel_N[i] = (alpha * n0_array[i] + beta * n1_array[i] + gamma * n2_array[i]).item()

                        #if(outline==False):
                        if(backFace):
                            #Line Art
                            imL.putpixel((x, -y),(0,0,0))
                            #toon Shade outline
                            imT.putpixel((x, -y),(0,0,0))
                            
                        else:
                            #Line Art
                            imL.putpixel((x, -y),(255,255,255))

                            # Toon Shading
                            cp_t = toonShading(camera, material_data, [x0,y0,z0], [x1,y1,z1],[x2,y2,z2], light_data, [[pixel_N[0]], [pixel_N[1]], [pixel_N[2]], [1]])
                            #stencilBuffer[x][y]= cp_t
                            imT.putpixel((x, -y),
                                    (round(np.nan_to_num(cp_t[0])), round(np.nan_to_num(cp_t[1])),
                                    round(np.nan_to_num(cp_t[2]))))

                        ## silhouette outline using 2-pass stencil buffer ##
                        #elif(outline==True):
                         #   cp_t =[0,0,0]
                          #  if(stencilBuffer[x][y]==[-1,-1,-1]):
                                #im.putpixel( (x,-y), (0,0,0))
                           # else:
                            #    cp_t = stencilBuffer[x][y]
                                # toon shade
                             #   im.putpixel((x, -y),
                              #      (round(np.nan_to_num(cp_t[0])), round(np.nan_to_num(cp_t[1])),
                               #     round(np.nan_to_num(cp_t[2]))))
                        zbuffer[x, y] = z



def renderToonLine(camera, triangle_data, material_data, xres, yres,rotation_matrix, scale_matrix, translate_matrix, mvp_matrix, normal_transformation_matrix, light_data):
    print("===================\nToon Shading and Line Art Begin\n")

    global imT, imL, width, height, zbuffer, stencilBuffer
    imT = Image.new('RGB', [xres, yres], 0x000000)
    imL = Image.new('RGB', [xres, yres], 0x000000)
    width, height = imT.size
    zbuffer = np.matrix(np.ones((xres, yres)) * np.inf)

    global noise
    noise = PerlinNoise()

    ## object silhouette outline using 2-pass stencil buffer ##
    #stencilBuffer = [ [[-1,-1,-1]]*width for i in range(height)]

    #global outline, outlineSize
    #outline = False
    #outlineSize=0.05

    #for i in range(2):
    renderGeom(camera, triangle_data, material_data, xres, yres,rotation_matrix, scale_matrix, translate_matrix, mvp_matrix, normal_transformation_matrix, light_data)
        #outline=True

    imT.show()
    imL.show()
    print("Toon Shading and Line Art Successful\n===================\n")