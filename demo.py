"""
Module for demonstrating the capabilities of the renderer.
"""

from render import Renderer


if __name__ == "__main__":
    renderer = Renderer("table_scene.json")
    ENABLE_SHADOWS = True
    ENABLE_AO = True
    CEL_SHADE = False
    HALFTONE_SHADE = False
    LINE_ART = False
    WIREFRAME = False
    RENDER_GEOMETRY_BUFFER = True
    RENDER_SHADOW_BUFFERS = True

    print("#####################################################")
    print("## AO - SHADOWS - NORMAL MAPS RENDER               ##")
    print("#####################################################")
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
