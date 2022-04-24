"""
Module for demonstrating the capabilities of the renderer.
"""

from argparse import ArgumentParser
from numpy import np
from render import Renderer


if __name__ == "__main__":
    parser = ArgumentParser(description="A program to demonstrate the capabilities of the renderer. Renders 6 images. "
                                        "The first image has not special features. The second image has shadows, "
                                        "ambient occlusion and normal mapping turned on. The third image is cel "
                                        "shaded. The fourth image is halftone shaded. The fifth image has line art "
                                        "rendering. The last image uses wireframe rendering.")
    parser.add_argument("-s", "--scene", help="Scene file path.", type=str, default="scenes/backpack_scene.json")
    args = parser.parse_args()

    np.seterr(divide='ignore', invalid='ignore')
    renderer = Renderer(args.scene)
    ENABLE_SHADOWS = False
    ENABLE_AO = False
    CEL_SHADE = False
    HALFTONE_SHADE = False
    LINE_ART = False
    WIREFRAME = False

    print("#####################################################")
    print("## DEFAULT (NO SPECIAL FEATUERS) RENDER            ##")
    print("#####################################################")
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()

    print("#####################################################")
    print("## AO - SHADOWS RENDER                             ##")
    print("#####################################################")
    ENABLE_SHADOWS = True
    ENABLE_AO = True
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()

    print("#####################################################")
    print("## CEL SHADED RENDER                               ##")
    print("#####################################################")
    CEL_SHADE = True
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()

    print("#####################################################")
    print("## HALFTONES RENDER                                ##")
    print("#####################################################")
    CEL_SHADE = False
    HALFTONE_SHADE = True
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()

    print("#####################################################")
    print("## LINE ART RENDER                                 ##")
    print("#####################################################")
    ENABLE_AO = False
    ENABLE_SHADOWS = False
    HALFTONE_SHADE = False
    LINE_ART = True
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()

    print("#####################################################")
    print("## WIREFRAME RENDER                                ##")
    print("#####################################################")
    LINE_ART = False
    WIREFRAME = True
    image = renderer.render(enable_shadows=ENABLE_SHADOWS, enable_ao=ENABLE_AO, cel_shade=CEL_SHADE,
                            halftone_shade=HALFTONE_SHADE, wireframe=WIREFRAME, line_art=LINE_ART)
    image.show()
