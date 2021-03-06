import math
import os

import numpy as np
from PIL import Image, ImageOps
import json
from shadow_renderer import *
from plane_renderer import *
from toon_shading import *
from wireframe_renderer import *
from halftone_render import *
from normal_map import *

# Functions
# Light calculation function
def light_calc(N):
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
def f01(xfun, yfun):
        return (y0 - y1) * xfun + (x1 - x0) * yfun + x0 * y1 - x1 * y0


def f12(xfun, yfun):
        return (y1 - y2) * xfun + (x2 - x1) * yfun + x1 * y2 - x2 * y1


def f20(xfun, yfun):
        return (y2 - y0) * xfun + (x0 - x2) * yfun + x2 * y0 - x0 * y2

# Save to PPM format function
def saveToPPM():
    # Saving to PPM  format
    if os.path.exists("hw5_gouraud.ppm"):
        os.remove("hw5_gouraud.ppm")

    if os.path.exists("hw5_phong.ppm"):
        os.remove("hw5_phong.ppm")

    ppmfile = open("hw5_gouraud.ppm", "a")
    ppmfile2 = open("hw5_phong.ppm", "a")

    ppmfile.write("P3 \n")
    ppmfile2.write("P3 \n")
    ppmfile.write(str(xres) + " " + str(yres) + "\n")
    ppmfile2.write(str(xres) + " " + str(yres) + "\n")
    ppmfile.write(str(max_RGB) + "\n")
    ppmfile2.write(str(max_RGB) + "\n")
    pixelString = ""
    pixelString2 = ""
    for y in range(xres):
        for x in range(yres):
            coord = x, y
            pixel = list(im.getpixel(coord))
            pixel2 = list(im2.getpixel(coord))
            if x == 0:
                pixelString = pixelString + str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2])
                pixelString2 = pixelString2 + str(pixel2[0]) + " " + str(pixel2[1]) + " " + str(pixel2[2])
            else:
                pixelString = pixelString + " " + str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2])
                pixelString2 = pixelString2 + " " + str(pixel2[0]) + " " + str(pixel2[1]) + " " + str(pixel2[2])

        pixelString = pixelString + "\n"
        pixelString2 = pixelString2 + "\n"

    ppmfile.write(pixelString)
    ppmfile.close()

    ppmfile2.write(pixelString2)
    ppmfile2.close()

    print("Saved to PPM format")


# Main
# Reading scene JSON
with open("scenes/teapot_scene.json") as json_file:
    scene_data = json.load(json_file)

# Getting camera data
camera_data = scene_data.get('scene').get('camera')
from_array = camera_data.get('from')
to_array = camera_data.get('to')
bounds_array = camera_data.get('bounds')
resolution_array = camera_data.get('resolution')

# Setting up Viewer
# Screen Size
xres = resolution_array[0]
yres = resolution_array[1]

im = Image.new('RGB', [xres, yres], 0x000000)
im2 = Image.new('RGB', [xres, yres], 0x000000)
width, height = im.size
max_RGB = 255

# Opening and saving texture map
texmap = Image.open("models/teapot/diffuse.jpeg")
texmap = ImageOps.mirror(texmap)
tex_xres, tex_yres = texmap.size

# Setting up background
for y in range(height):
    for x in range(width):
        im.putpixel((x, y), (128, 128, 128))
        im2.putpixel((x, y), (128, 128, 128))

# Constructing camera matrix
camera_n = [0, 0, 0]
camera_v = [0, 1, 0]  # fake V
camera_u = [0, 0, 0]
camera_r = from_array
for i in range(0, len(camera_r)):
    camera_n[i] = camera_r[i] - to_array[i]
    #camera_v[i] = to_array[i] - from_array[i]

camera_n = camera_n / np.linalg.norm(camera_n)
camera_u = np.cross(camera_v, camera_n)
camera_u = camera_u / np.linalg.norm(camera_u)
camera_v = np.cross(camera_n, camera_u)

camera_matrix = [[camera_u[0], camera_u[1], camera_u[2], -np.dot(camera_r, camera_u)],
                 [camera_v[0], camera_v[1], camera_v[2], -np.dot(camera_r, camera_v)],
                 [camera_n[0], camera_n[1], camera_n[2], -np.dot(camera_r, camera_n)],
                 [0, 0, 0, 1]]

# Constructing perspective matrix
# Perspective matrix
near = bounds_array[0]
far = bounds_array[1]
right = bounds_array[2]
left = bounds_array[3]
top = bounds_array[4]
bottom = bounds_array[5]

perspective_matrix = [[2 * near / (right - left), 0, (right + left) / (right - left), 0],
                      [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
                      [0, 0, -((far + near) / (far - near)), -((2 * far * near) / (far - near))],
                      [0, 0, -1, 0]]

# Getting light data
light_data = scene_data.get('scene').get('lights')
for j in range(0, len(light_data)):
    if light_data[j].get('type') == 'ambient':
        la = light_data[j].get('color')
        la_intensity = light_data[j].get('intensity')
    else:
        ld = light_data[j].get('color')
        ld_intensity = light_data[j].get('intensity')
        ld_from = light_data[j].get('from')
        ld_to = light_data[j].get('to')

# Converting directional light coord
light_n = [0, 0, 0, 1]
light_v = [0, 1, 0]  # fake V
light_u = [0, 0, 0]
light_r = ld_from

for i in range(0, len(light_r)):
    light_n[i] = light_r[i] - ld_to[i]

# Creating light matrix
light_n = light_n[:-1]
light_n = light_n / np.linalg.norm(light_n)

light_u = np.cross(light_v, light_n)
light_u = light_u / np.linalg.norm(light_u)
light_v = np.cross(light_n, light_u)

light_matrix = [[light_u[0], light_u[1], light_u[2], -np.dot(light_r, light_u)],
                [light_v[0], light_v[1], light_v[2], -np.dot(light_r, light_v)],
                [light_n[0], light_n[1], light_n[2], -np.dot(light_r, light_n)],
                [0, 0, 0, 1]]

# Creating light perspective matrix
light_near = 3
light_far = 10
light_right = 1
light_left = -1
light_top = 1
light_bottom = -1

light_perspective_matrix = [
    [2 * light_near / (light_right - light_left), 0, (light_right + light_left) / (light_right - light_left), 0],
    [0, 2 * light_near / (light_top - light_bottom), (light_top + light_bottom) / (light_top - light_bottom), 0],
    [0, 0, -((light_far + light_near) / (light_far - light_near)),
     -((2 * light_far * light_near) / (light_far - light_near))],
    [0, 0, -1, 0]]


E = camera_n

# Getting shape data
# Getting material datax
x = 2
material_data = scene_data.get('scene').get('shapes')[x].get('material')
Cs = material_data.get("Cs")
Ka = material_data.get("Ka")
Kd = material_data.get("Kd")
Ks = material_data.get("Ks")
s = material_data.get("n")

#Cs = [1,1,1]

Ka = 1.0
Kd = 1.0
Ks = 1.5


# Getting transform data
transform_data = scene_data.get('scene').get('shapes')[x].get('transforms')
# Rotation Data
if "Ry" in transform_data[0]:
    theta = transform_data[0].get("Ry") * math.pi / 180
    rotation_matrix = [[math.cos(theta), 0, math.sin(theta), 0],
                       [0, 1, 0, 0],
                       [-math.sin(theta), 0, math.cos(theta), 0],
                       [0, 0, 0, 1]]
if "Rx" in transform_data[0]:
    theta = transform_data[0].get("Rx")
    rotation_matrix = [[1, 0, 0, 0],
                       [0, math.cos(theta), -math.sin(theta), 0],
                       [0, math.sin(theta), math.cos(theta), 0],
                       [0, 0, 0, 1]]
if "Rz" in transform_data[0]:
    theta = transform_data[0].get("Rz")
    rotation_matrix = [[math.cos(theta), -math.sin(theta), 0, 0],
                       [math.sin(theta), math.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]
# Scale Data
scale_array = transform_data[1].get("S")

scale_matrix = [[scale_array[0], 0, 0, 0],
                [0, scale_array[1], 0, 0],
                [0, 0, scale_array[2], 0],
                [0, 0, 0, 1]]

scale_matrix_inverse = [[1 / scale_array[0], 0, 0, 0],
                        [0, 1 / scale_array[1], 0, 0],
                        [0, 0, 1 / scale_array[2], 0],
                        [0, 0, 0, 1]]

scale_matrix_inverse_transpose = np.transpose(scale_matrix_inverse)

# Translation Data
translate_array = transform_data[2].get("T")
translate_matrix = [[1, 0, 0, translate_array[0]],
                    [0, 1, 0, translate_array[1]],
                    [0, 0, 1, translate_array[2]],
                    [0, 0, 0, 1]]

# Concatenated Model-View-Projection (Camera) Matrix
mvp_matrix = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
mvp_matrix = np.matmul(rotation_matrix, mvp_matrix)
mvp_matrix = np.matmul(scale_matrix, mvp_matrix)
mvp_matrix = np.matmul(translate_matrix, mvp_matrix)
mvp_matrix = np.matmul(camera_matrix, mvp_matrix)
mvp_matrix = np.matmul(perspective_matrix, mvp_matrix)

# Concatenated Model-View-Projection (Light) Matrix
light_mvp_matrix = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
light_mvp_matrix = np.matmul(rotation_matrix, light_mvp_matrix)
light_mvp_matrix = np.matmul(scale_matrix, light_mvp_matrix)
light_mvp_matrix = np.matmul(translate_matrix, light_mvp_matrix)
light_mvp_matrix = np.matmul(light_matrix, light_mvp_matrix)
light_mvp_matrix = np.matmul(light_perspective_matrix, light_mvp_matrix)

# Concatenated Plane Model-View-Projection (Camera) Matrix
plane_mvp_matrix  = [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
plane_mvp_matrix = np.matmul(camera_matrix, plane_mvp_matrix)
plane_mvp_matrix = np.matmul(perspective_matrix, plane_mvp_matrix)

# Concatenated Plane Model-View-Projection (Light) Matrix
plane_light_mvp_matrix  = [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]
plane_light_mvp_matrix = np.matmul(light_matrix, plane_light_mvp_matrix)
plane_light_mvp_matrix = np.matmul(light_perspective_matrix, plane_light_mvp_matrix)

# Concatenated Transformation Matrix For Normals
normal_transformation_matrix = [[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]
normal_transformation_matrix = np.matmul(scale_matrix_inverse_transpose, normal_transformation_matrix)
normal_transformation_matrix = np.matmul(rotation_matrix, normal_transformation_matrix)

# Opening geometry file
geo_file_name = scene_data.get('scene').get('shapes')[x].get('geometry') + ".json"
with open(geo_file_name) as json_file:
    triangle_data = json.load(json_file)

# Setting up z-buffer
zbuffer = np.matrix(np.ones((xres, yres)) * np.inf)
shadow_buffer = np.matrix(np.ones((xres, yres)) * np.inf)

# Creating shadow buffer
createShadowBuffer(triangle_data.get('data'), xres, yres, light_mvp_matrix, shadow_buffer)

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
    c1 = light_calc(n0_array)
    c2 = light_calc(n1_array)
    c3 = light_calc(n2_array)

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
                        # Gouraud Shading - interpolating color
                        cp_g[i] = ((alpha * c1[i] + beta * c2[i] + gamma * c3[i]).item())

                        if cp_g[i] > 1.0:
                            cp_g[i] = 1.0
                        # Phong Shading - interpolating normal
                        pixel_N[i] = (alpha * n0_array[i] + beta * n1_array[i] + gamma * n2_array[i]).item()

                    # Phong Shading
                    cp_ph = light_calc([[pixel_N[0]], [pixel_N[1]], [pixel_N[2]], [1]])

                    # Gouraud
                    im2.putpixel((x, -y),
                                (round((cp_g[0]) * pOutputRGB[0]*Kt), round((cp_g[1]) * pOutputRGB[1]*Kt),
                                 round((cp_g[2]) * pOutputRGB[2]*Kt)))
                    # Phong
                    im.putpixel((x, -y),
                                 (round((cp_ph[0]) * pOutputRGB[0]*Kt),
                                  round((cp_ph[1]) * pOutputRGB[1]*Kt),
                                  round((cp_ph[2]) * pOutputRGB[2]*Kt)))
                    zbuffer[x, y] = z


# Rendering x-z plane
renderPlane(im, xres, yres, zbuffer, plane_mvp_matrix, light_n, la, la_intensity, ld, ld_intensity, E, Ks, Kd, Ka, s)
# im4 = renderPlane(im, xres, yres, zbuffer, light_matrix, light_perspective_matrix)

with open("models/plane/plane.json") as json_file:
    plane_data = json.load(json_file)
renderShadow(im, plane_data, xres, yres, plane_mvp_matrix, plane_light_mvp_matrix, shadow_buffer, zbuffer, la, la_intensity, Ka)
renderToonLine(camera_data, triangle_data, material_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, mvp_matrix, normal_transformation_matrix, light_data)
renderWireframe(triangle_data, xres, yres, mvp_matrix, normal_transformation_matrix)
renderHalftone(camera_data, triangle_data, material_data, xres, yres, rotation_matrix, scale_matrix, translate_matrix, scale_matrix_inverse_transpose, camera_matrix, perspective_matrix, light_data)

depth_map = generate_depth_map(triangle_data, plane_data, mvp_matrix, xres, yres)
normal_map_im = generate_normal_map(depth_map, xres, yres)
render_with_normal_map(normal_map_im, triangle_data, mvp_matrix, normal_transformation_matrix, xres, yres, texmap, tex_xres, tex_yres, light_n, la, la_intensity, ld, ld_intensity, Ka, Kd, Ks, E, s)

im.show()
#im2.show()
#saveToPPM()
