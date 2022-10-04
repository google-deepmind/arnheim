# Generative Art Using Neural Visual Grammars and Dual Encoders

## Arnheim 1

The original algorithm from the paper
[Generative Art Using Neural Visual Grammars and Dual Encoders](https://arxiv.org/abs/2105.00162)
running on 1 GPU allows optimization of any image using a genetic algorithm.
This is much more general but much slower than using Arnheim 2 which uses
gradients.

## Arnheim 2

A reimplementation of the Arnheim 1 generative architecture in the CLIPDraw
framework allowing optimization of its parameters using gradients. Much more
efficient than Arnheim 1 above but requires differentiating through the image
itself.

## Arnheim 3 (aka CLIP-CLOP: CLIP-Guided Collage and Photomontage)

A spatial transformer-based Arnheim implementation for generating collage images.
It employs a combination of evolution and training to create collages from
opaque to transparent image patches. 

Example patch datasets, with the exception of 'Fruit and veg', are provided under
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).
The 'Fruit and veg' patches in `collage_patches/fruit.npy` are based on a subset 
of the Kaggle Fruits 360 and are provided under
[CC BY-SA 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/), 
as are all example collages using them. 

![The Fall of the Damned by Rubens and Eaton.](https://raw.githubusercontent.com/deepmind/arnheim/main/images/fall_of_the_damned.jpg)
![Collages made of different numbers of tree leaves patches (bulls in the top row), as well as Degas-inspired ballet dancers made from animals, faces made of fruit and still life or landscape made from patches of animals.](https://raw.githubusercontent.com/deepmind/arnheim/main/images/bulls_ballet_faces_nature.jpg)

## Usage

Usage instructions are included in the Colabs which open and run on the
free-to-use Google Colab platform - just click the buttons below! Improved
performance and longer timeouts are available with Colab Pro.

Arnheim 1 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/main/arnheim_1.ipynb)

Arnheim 2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/main/arnheim_2.ipynb)

Arnheim 3 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/main/arnheim_3.ipynb)

Arnheim 3 Patch Maker [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/main/arnheim_3_patch_maker.ipynb)

## Video illustration of the CLIP-CLOP Collage and Photomontage Generator (Arnheim 3)

[![CLIP-CLOP Collage and Photomontage Generator](https://img.youtube.com/vi/VnO4tibP9cg/0.jpg)](https://youtu.be/VnO4tibP9cg)


## Citing this work

If you use this code (or any derived code), data or these models in your work,
please cite the relevant accompanying papers on [Generative Art Using Neural Visual Grammars and Dual Encoders](https://arxiv.org/abs/2105.00162)
or on [CLIP-CLOP: CLIP-Guided Collage and Photomontage](https://arxiv.org/abs/2205.03146).

```
@misc{fernando2021genart,
      title={Generative Art Using Neural Visual Grammars and Dual Encoders},
      author={Chrisantha Fernando and S. M. Ali Eslami and Jean-Baptiste Alayrac and Piotr Mirowski and Dylan Banarse and Simon Osindero}
      year={2021},
      eprint={2105.00162},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@inproceedings{mirowski2022clip,
               title={CLIP-CLOP: CLIP-Guided Collage and Photomontage},
               author={Piotr Mirowski and Dylan Banarse and Mateusz Malinowski and Simon Osindero and Chrisantha Fernando},
               booktitle={Proceedings of the Thirteenth International Conference on Computational Creativity},
               year={2022}
}
```

## Disclaimer

This is not an official Google product.

CLIPDraw provided under license, Copyright 2021 Kevin Frans.

Other works may be copyright of the authors of such work.
