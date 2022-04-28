# Scratch Renderer

This is a renderer built from scratch in Python without any graphics library calls (other than to set a pixel on an image). The renderer implements advanced lighting features such as screen space ambient occlusion, shadow mapping, normal mapping and stylized rendering effects such as cel shading, halftone shading, line art and wireframe rendering. This is built as the final project for the CSCI 580 class (Spring 2022) at the University of Southern California.

Shravan Kumar ([@ShravanK55](https://github.com/ShravanK55)) worked on ambient occlusion, Luiz Cilento ([@cilento1](https://github.com/cilento1)) worked on shadow mapping, Rashi Sinha ([@Rashi-Sinha](https://github.com/Rashi-Sinha)) worked on stylized rendering and Yaorong Xie ([@yaorongxie](https://github.com/yaorongxie)) worked on normal mapping.

# Usage

To run the renderer, run the render.py module.

    python render.py --scene="scenes/jinx_scene.json"

The arguments that can be provided in the command line are:

    usage: render.py [-h] [-s SCENE] [--disable_shadows] [--disable_ssao] [-a] [-c] [-t] [-l] [-w] [-d] [-f]

    A program that renders 3D scenes.

    options:
      -h, --help            show this help message and exit
      -s SCENE, --scene SCENE
                            Scene file path.
      --disable_shadows     Disable shadows in the renderer.
      --disable_ssao        Disable ambient occlusion in the renderer.
      -a, --anti_aliasing   Enable anti aliasing in the renderer.
      -c, --cel_shade       Turn on cel shading.
      -t, --halftone_shade  Turn on halftone shading.
      -l, --line_art        Turn on line art shading.
      -w, --wireframe       Turn on wireframe mode.
      -d, --draw_buffers    Draw the geometry and shadow buffers.
      -f, --write_to_file   Write the output images to files.

To run a focused demo with a comparison of every feature, run the demo.py module.

    python demo.py --scene="scenes/backpack_scene.json"

# Showcase

The section has images that were created using our renderer.
![Jinx](https://i.imgur.com/4uBACmd.png)
![Hollow Knight](https://i.imgur.com/9yTIJ7h.png)
![McLaren F1 Car](https://i.imgur.com/iY8Z0Nh.png)
![Viking Room](https://i.imgur.com/Sj1YXwL.png)
![Artorias](https://i.imgur.com/tSz0H1b.png)
![Ezio](https://i.imgur.com/bPu8DnL.png)
![Malenia](https://i.imgur.com/FCHBkrX.png)
![Luffy](https://i.imgur.com/oBP0tDq.png)
![Thousand Sunny](https://i.imgur.com/rO3uKR6.png)
![Backpacks](https://i.imgur.com/H6lAUIk.png)

# Credits

Professor [Saty Raghavachary](https://viterbi.usc.edu/directory/faculty/Raghavachary/Saty) from the University of Southern California: For his invaluable teaching and help in the CSCI-580 course.

To all model creators: For the incredible models used in the showcase. Source attributions are under the models folder.
