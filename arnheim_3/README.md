# Generative Art Using Neural Visual Grammars and Dual Encoders

## Arnheim 3 (Command line version)

A spatial transformer-based Arnheim implementation for generating collage images.
It employs a combination of evolution and training to create collages from
opaque to transparent image patches. Example patch datasets are provided under
[CC BY-SA 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/).
The 'Fruit and veg' patches are based on a subset of the
[Kaggle Fruits 360](https://www.kaggle.com/moltean/fruits) under the same
license.

## Installation

Clone this GitHub repository and go the `arnheim_3` directory:
```sh
git clone https://github.com/piotrmirowski/arnheim.git
cd arnheim/arnheim_3
```

Install the required Python libraries:
```sh
python3 -m pip install -r requirements.txt
```

Install [CLIP](https://github.com/openai/CLIP) from OpenAI's GitHub repository:
```sh
python3 -m pip install git+https://github.com/openai/CLIP.git --no-deps
```

When using GCP, it might help to enable remote desktop in both your local Chrome browser and on the GCP virtual machine, which can be done following [these instructions](https://cloud.google.com/architecture/chrome-desktop-remote-on-compute-engine#cinnamon).

## Usage

Configuration files are stored in YAML format in subdirectory `configs`. For instance script `configs/config_compositional_tiled.yaml` generates a composttional collage with global prompt `a photorealistic chicken` and 9 local prompts for `sky`, `sun`, `moon`, `tree`, `field` and `chicken`.

Please refer to `configs/config.yaml` and to the help for explanation about the config.
```sh
python3 main.py --help
```

To run with CUDA on a GPU accelerator:
```sh
python3 main.py --config configs/config_compositional.yaml
```

To run without CUDA (e.g., on Mac OS - note this will be considerably slower):
```sh
python3 main.py --no-cuda --config configs/config_compositional.yaml
```

By default, results are stored in a directory named `output_YYYYMMDD_hhmmss` (based on the timestamp) and contain the config `.yaml` file, and the resulting collage (and tiles) as `.png` and `.npy` files.

## Citing this work

If you use this code (or any derived code), data or these models in your work,
please cite the relevant accompanying [paper](https://arxiv.org/abs/2105.00162).

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

## Disclaimer

This is not an official Google product.

CLIPDraw provided under license, Copyright 2021 Kevin Frans.

Other works may be copyright of the authors of such work.
