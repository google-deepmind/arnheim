"""Arnheim 3 - Collage Creator
Piotr Mirowski, Dylan Banarse, Mateusz Malinowski, Yotam Doron, Oriol Vinyals,
Simon Osindero, Chrisantha Fernando
DeepMind, 2021-2022

Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import configargparse
from datetime import datetime
import glob
import os
import pathlib
import subprocess
import sys
import yaml

import numpy as np
import torch

import clip

import src.collage as collage
import src.video_utils as video_utils


# Specify (and override) the config.
ap = configargparse.ArgumentParser(default_config_files=["configs/config.yaml"])
ap.add_argument("-c", "--config", required=True, is_config_file=True,
                help="Config file")

# Use CUDA?
ap.add_argument("--cuda", dest="cuda", action="store_true")
ap.add_argument("--no-cuda", dest="cuda", action="store_false")
ap.set_defaults(cuda=True)
ap.add_argument("--torch_device", type=str, default="cuda",
                help="Alternative way of specifying the device: cuda or cpu?")

# Output directory.
ap.add_argument("--init_checkpoint", type=str, default="",
                help="Path to checkpoint")

# Output directory.
ap.add_argument("--output_dir", type=str, default="",
                help="Output directory")

# Clean-up?
ap.add_argument("--clean_up", dest='clean_up', help="Remove all working files",
                action='store_true')
ap.add_argument("--no-clean_up", dest='clean_up',
                help="Remove all working files", action='store_false')
ap.set_defaults(clean_up=False)

# GUI?
ap.add_argument('--gui', dest='gui', action='store_true')
ap.add_argument('--no-gui', dest='gui', action='store_false')
ap.set_defaults(gui=False)

# Video and tracing.
ap.add_argument("--video_steps", type=int, default=0,
                help="Number of steps between two video frames")
ap.add_argument("--trace_every", type=int, default=50,
                help="Number of steps between two logging traces")
ap.add_argument('--population_video', dest='population_video',
                action='store_true', help='Write the video of population?')
ap.add_argument('--no-population_video', dest='population_video',
                action='store_false', help='Write the video of population?')
ap.set_defaults(population_video=False)

# Canvas size.
ap.add_argument("--canvas_width", type=int, default=224,
                help="Image width for CLIP optimization")
ap.add_argument("--canvas_height", type=int, default=224,
                help="Image height for CLIP optimization")
ap.add_argument("--max_block_size_high_res", type=int, default=2000,
                help="Max block size for high-res image")

# Render methods.
ap.add_argument("--render_method", type=str, default="transparency",
                help="opacity patches overlay each other using combinations of "
                "alpha and depth, transparency _adds_ patch RGB values (black "
                "therefore appearing transparent), masked_transparency_clipped "
                "and masked_transparency_normed blend patches using the alpha "
                "channel")
ap.add_argument("--num_patches", type=int, default=100,
                help="Number of patches")
ap.add_argument("--colour_transformations", type=str, default="RGB space",
                help="Can be none, RGB space or HHSV space")
ap.add_argument("--invert_colours", dest="invert_colours", action='store_true',
                help="Invert image colours to have a white background?")
ap.add_argument("--no-invert_colours", dest="invert_colours",
                action='store_false',
                help="Invert image colours to have a white background?")
ap.set_defaults(invert_colours=False)
ap.add_argument("--high_res_multiplier", type=int, default=4,
                help="Ratio between large canvas and CLIP-optimized canvas")
ap.add_argument('--save_all_arrays', dest='save_all_arrays',
                action='store_true',
                help='Save the optimised patch arrays as an npy file?')
ap.add_argument('--no-save_all_arrays', dest='save_all_arrays',
                action='store_false',
                help='Save the optimised patch arrays as an npy file?')
ap.set_defaults(save_all_arrays=False)

# Affine transform settings.
ap.add_argument("--min_trans", type=float, default=-1.,
                help="Translation min for X and Y")
ap.add_argument("--max_trans", type=float, default=1.,
                help="Translation max for X and Y")
ap.add_argument("--min_trans_init", type=float, default=-1.,
                help="Initial translation min for X and Y")
ap.add_argument("--max_trans_init", type=float, default=1.,
                help="Initial translation max for X and Y")
ap.add_argument("--min_scale", type=float, default=1.,
                help="Scale min (> 1 means zoom out and < 1 means zoom in)")
ap.add_argument("--max_scale", type=float, default=2.,
                help="Scale max (> 1 means zoom out and < 1 means zoom in)")
ap.add_argument("--min_squeeze", type=float, default=0.5,
                help="Min ratio between X and Y scale")
ap.add_argument("--max_squeeze", type=float, default=2.,
                help="Max ratio between X and Y scale")
ap.add_argument("--min_shear", type=float, default=-0.2,
                help="Min shear deformation")
ap.add_argument("--max_shear", type=float, default=0.2,
                help="Max shear deformation")
ap.add_argument("--min_rot_deg", type=float, default=-180, help="Min rotation")
ap.add_argument("--max_rot_deg", type=float, default=180, help="Max rotation")

# Colour transform settings.
ap.add_argument("--min_rgb", type=float, default=-0.2,
                help="Min RGB between -1 and 1")
ap.add_argument("--max_rgb", type=float, default=1.0,
                help="Max RGB between -1 and 1")
ap.add_argument("--initial_min_rgb", type=float, default=0.5,
                help="Initial min RGB between -1 and 1")
ap.add_argument("--initial_max_rgb", type=float, default=1.,
                help="Initial max RGB between -1 and 1")
ap.add_argument("--min_hue_deg", type=float, default=0.,
                help="Min hue between 0 and 360")
ap.add_argument("--max_hue_deg", type=float, default=360,
                help="Max hue (in degrees) between 0 and 360")
ap.add_argument("--min_sat", type=float, default=0,
                help="Min saturation between 0 and 1")
ap.add_argument("--max_sat", type=float, default=1,
                help="Max saturation between 0 and 1")
ap.add_argument("--min_val", type=float, default=0,
                help="Min value between 0 and 1")
ap.add_argument("--max_val", type=float, default=1,
                help="Max value between 0 and 1")

# Training settings.
ap.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model")
ap.add_argument("--optim_steps", type=int, default=10000,
                help="Number of training steps (between 0 and 20000)")
ap.add_argument("--learning_rate", type=float, default=0.1,
                help="Learning rate, typically between 0.05 and 0.3")
ap.add_argument("--use_image_augmentations", dest="use_image_augmentations",
                action='store_true',
                help="User image augmentations for CLIP evaluation?")
ap.add_argument("--no-use_image_augmentations", dest="use_image_augmentations",
                action='store_false',
                help="User image augmentations for CLIP evaluation?")
ap.set_defaults(use_image_augmentations=True)
ap.add_argument("--num_augs", type=int, default=4,
                help="Number of image augmentations to use in CLIP evaluation")
ap.add_argument("--use_normalized_clip", dest="use_normalized_clip",
                action='store_true',
                help="Normalize colours for CLIP, generally leave this as True")
ap.add_argument("--no-use_normalized_clip", dest="use_normalized_clip",
                action='store_false',
                help="Normalize colours for CLIP, generally leave this as True")
ap.set_defaults(use_normalized_clip=False)
ap.add_argument("--gradient_clipping", type=float, default=10.0,
                help="Gradient clipping during optimisation")
ap.add_argument("--initial_search_size", type=int, default=1,
                help="Initial random search size (1 means no search)")
ap.add_argument("--initial_search_num_steps", type=int, default=1,
                help="Number of gradient steps in initial random search size "
                "(1 means only random search, more means gradient descent)")

# Evolution settings.
ap.add_argument("--pop_size", type=int, default=2,
                help="For evolution set this to greater than 1")
ap.add_argument("--evolution_frequency", type=int, default= 100,
                help="Number of gradient steps between two evolution mutations")
ap.add_argument("--ga_method", type=str, default="Microbial",
                help="Microbial: loser of randomly selected pair is replaced "
                "by mutated winner. A low selection pressure. Evolutionary "
                "Strategies: mutantions of the best individual replace the "
                "rest of the population. Much higher selection pressure than "
                "Microbial GA")

# Mutation levels.
ap.add_argument("--pos_and_rot_mutation_scale", type=float, default=0.02,
                help="Probability of position and rotation mutations")
ap.add_argument("--scale_mutation_scale", type=float, default=0.02,
                help="Probability of scale mutations")
ap.add_argument("--distort_mutation_scale", type=float, default=0.02,
                help="Probability of distortion mutations")
ap.add_argument("--colour_mutation_scale", type=float, default=0.02,
                help="Probability of colour mutations")
ap.add_argument("--patch_mutation_probability", type=float, default=1,
                help="Probability of patch mutations")

# Visualisation.
ap.add_argument("--max_multiple_visualizations", type=int, default=5,
                help="Limit the number of individuals shown during training")

# Load segmented patches.
ap.add_argument("--multiple_patch_set", default=None,
                action='append', dest="multiple_patch_set")
ap.add_argument("--multiple_fixed_scale_patches", default=None,
                action='append', dest="multiple_fixed_scale_patches")
ap.add_argument("--multiple_patch_max_proportion", default=None,
                action='append', dest="multiple_patch_max_proportion")
ap.add_argument("--multiple_fixed_scale_coeff", default=None,
                action='append', dest="multiple_fixed_scale_coeff")
ap.add_argument("--patch_set", type=str, default="animals.npy",
                help="Name of Numpy file with patches")
ap.add_argument("--patch_repo_root", type=str,
                default=
                "https://storage.googleapis.com/dm_arnheim_3_assets/collage_patches",
                help="URL to patches")
ap.add_argument("--url_to_patch_file", type=str, default="",
                help="URL to a patch file")

# Resize image patches to low- and high-res.
ap.add_argument("--fixed_scale_patches", dest="fixed_scale_patches",
                action='store_true', help="Use fixed scale patches?")
ap.add_argument("--no-fixed_scale_patches", dest="fixed_scale_patches",
                action='store_false', help="Use fixed scale patches?")
ap.set_defaults(fixed_scale_patches=True)
ap.add_argument("--fixed_scale_coeff", type=float, default=0.7,
                help="Scale coeff for fixed scale patches")
ap.add_argument("--normalize_patch_brightness",
                dest="normalize_patch_brightness", action='store_true',
                help="Normalize the brightness of patches?")
ap.add_argument("--no-normalize_patch_brightness",
                dest="normalize_patch_brightness", action='store_false',
                help="Normalize the brightness of patches?")
ap.set_defaults(normalize_patch_brightness=False)
ap.add_argument("--patch_max_proportion", type=int, default= 5,
                help="Max proportion of patches, between 2 and 8")
ap.add_argument("--patch_width_min", type=int, default=16,
                help="Min width of patches")
ap.add_argument("--patch_height_min", type=int, default=16,
                help="Min height of patches")

# Configure a background, e.g. uploaded picture or solid colour.
ap.add_argument("--background_use", type=str, default="Global",
                help="Global: use image across whole image, "
                "or Local: reuse same image for every tile")
ap.add_argument("--background_url", type=str, default="",
                help="URL for background image")
ap.add_argument("--background_red", type=int, default=0,
                help="Red solid colour background (0 to 255)")
ap.add_argument("--background_green", type=int, default=0,
                help="Green solid colour background (0 to 255)")
ap.add_argument("--background_blue", type=int, default=0,
                help="Blue solid colour background (0 to 255)")

# Configure image prompt and content.
ap.add_argument("--global_prompt", type=str,
                default="Roman mosaic of an unswept floor",
                help="Global description of the image")

# Tile prompts and tiling settings.
ap.add_argument("--tile_images", action='store_true', dest="tile_images",
                help="Tile images?")
ap.add_argument("--no-tile_images", action='store_false', dest="tile_images",
                help="Tile images?")
ap.set_defaults(tile_images=False)
ap.add_argument("--tiles_wide", type=int, default=1,
                help="Number of width tiles")
ap.add_argument("--tiles_high", type=int, default=1,
                help="Number of height tiles")
ap.add_argument("--global_tile_prompt", dest="global_tile_prompt",
                action='store_true',
                help="Global tile prompt uses global_prompt (previous cell) "
                "for *all* tiles (e.g. Roman mosaic of an unswept floor)")
ap.add_argument("--no-global_tile_prompt", dest="global_tile_prompt",
                action='store_false',
                help="Global tile prompt uses global_prompt (previous cell) "
                "for *all* tiles (e.g. Roman mosaic of an unswept floor)")
ap.set_defaults(global_tile_prompt=False)
ap.add_argument("--tile_prompt_string", type=str, default="",
                help="Otherwise, specify multiple tile prompts with columns "
                "separated by | and / to delineate new row. E.g. multiple "
                "prompts for a 3x2 'landscape' image: "
                "'sun | clouds | sky / fields | fields | trees'")

# Composition prompts.
ap.add_argument("--compositional_image", dest="compositional_image",
                action="store_true",
                help="Use additional prompts for different regions")
ap.add_argument("--no-compositional_image", dest="compositional_image",
                action="store_false",
                help="Do not use additional prompts for different regions")
ap.set_defaults(compositional_image=False)
# Single image (i.e. no tiling) composition prompts:
# specify 3x3 prompts for each composition region.
ap.add_argument("--prompt_x0_y0", type=str,
                default="a photorealistic sky with sun", help="Top left prompt")
ap.add_argument("--prompt_x1_y0", type=str,
                default="a photorealistic sky", help="Top centre prompt")
ap.add_argument("--prompt_x2_y0", type=str,
                default="a photorealistic sky with moon", help="Top right prompt")
ap.add_argument("--prompt_x0_y1", type=str,
                default="a photorealistic tree", help="Middle left prompt")
ap.add_argument("--prompt_x1_y1", type=str,
                default="a photorealistic tree", help="Centre prompt")
ap.add_argument("--prompt_x2_y1", type=str,
                default="a photorealistic tree", help="Middle right prompt")
ap.add_argument("--prompt_x0_y2", type=str,
                default="a photorealistic field", help="Bottom left prompt")
ap.add_argument("--prompt_x1_y2", type=str,
                default="a photorealistic field", help="Bottom centre prompt")
ap.add_argument("--prompt_x2_y2", type=str,
                default="a photorealistic chicken", help="Bottom right prompt")

# Tile composition prompts.
ap.add_argument("--tile_prompt_formating", type=str, default="close-up of {}",
                help="This string is formated to autogenerate region prompts "
                "from tile prompt. e.g. close-up of {}")

# Get the config.
config = vars(ap.parse_args())

print(config)

# Adjust config for compositional image.
if config["compositional_image"] == True:
  print("Generating compositional image")
  config['canvas_width'] *= 2
  config['canvas_height'] *= 2
  config['high_res_multiplier'] = int(config['high_res_multiplier'] / 2)
  print("Using one image augmentations for compositional image creation.")
  config["use_image_augmentations"] = True
  config["num_augs"] = 1

# Turn off tiling if either boolean is set or width/height set to 1.
if (not config["tile_images"] or
    (config["tiles_wide"] == 1 and config["tiles_high"] == 1)):
  print("No tiling.")
  config["tiles_wide"] = 1
  config["tiles_high"] = 1
  config["tile_images"] = False

# Default output dir.
if len(config["output_dir"]) == 0:
  config["output_dir"] = "output_"
  config["output_dir"] += datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
  config["output_dir"] += '/'

# Print the config.
print("\n")
yaml.dump(config, sys.stdout, default_flow_style=False, allow_unicode=True)
print("\n\n")


# Configure CUDA.
print("Torch version:", torch.__version__)
if not config["cuda"] or config["torch_device"] == "cpu":
  config["torch_device"] = "cpu"
  config["cuda"] = False
device = torch.device(config["torch_device"])

# Configure ffmpeg.
os.environ["FFMPEG_BINARY"] = "ffmpeg"


# Initialise and load CLIP model.
print(f"Downloading CLIP model {config['clip_model']}...")
clip_model, _ = clip.load(config["clip_model"], device, jit=False)

# Make output dir.
output_dir = config["output_dir"]
print(f"Storing results in {output_dir}\n")
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Save the config.
config_filename = config["output_dir"] + '/' + "config.yaml"
with open(config_filename, "w") as f:
  yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# Tiling.
if not config["tile_images"] or config["global_tile_prompt"]:
  tile_prompts = (
    [config["global_prompt"]] * config["tiles_high"] * config["tiles_wide"])
else:
  tile_prompts = []
  count_y = 0
  count_x = 0
  for row in config["tile_prompt_string"].split("/"):
    for prompt in row.split("|"):
      prompt = prompt.strip()
      tile_prompts.append(prompt)
      count_x += 1
    if count_x != config["tiles_wide"]:
      w = config["tiles_wide"]
      raise ValueError(
        f"Insufficient prompts for row {count_y}; expected {w}, got {count_x}")
    count_x = 0
    count_y += 1
  if count_y != config["tiles_high"]:
    h = config["tiles_high"]
    raise ValueError(f"Insufficient prompt rows; expected {h}, got {count_y}")

print("Tile prompts: ", tile_prompts)
# Prepare duplicates of config data if required for tiles.
tile_count = 0
all_prompts = []
for y in range(config["tiles_high"]):
  for x in range(config["tiles_wide"]):
    list_tile_prompts = []
    if config["compositional_image"]:
      if config["tile_images"]:
        list_tile_prompts = [
            config["tile_prompt_formating"].format(tile_prompts[tile_count])
            ] * 9
      else:
        list_tile_prompts = [
            config["prompt_x0_y0"], config["prompt_x1_y0"],
            config["prompt_x2_y0"],
            config["prompt_x0_y1"], config["prompt_x1_y1"],
            config["prompt_x2_y1"],
            config["prompt_x0_y2"], config["prompt_x1_y2"],
            config["prompt_x2_y2"]]
    list_tile_prompts.append(tile_prompts[tile_count])
    tile_count += 1
    all_prompts.append(list_tile_prompts)
print(f"All prompts: {all_prompts}")


# Background.
background_image = None
background_url = config["background_url"]
if len(background_url) > 0:
  # Load background image from URL.
  if background_url.startswith("http"):
    background_image = video_utils.cached_url_download(background_url,
                                                       format="image_as_np")
  else:
    background_image = video_utils.load_image(background_url,
                                              show=config["gui"])
else:
  background_image = np.ones((10, 10, 3), dtype=np.float32)
  background_image[:, :, 0] = config["background_red"] / 255.
  background_image[:, :, 1] = config["background_green"] / 255.
  background_image[:, :, 2] = config["background_blue"] / 255.
  print('Defined background colour ({}, {}, {})'.format(
      config["background_red"], config["background_green"],
      config["background_blue"]))


# Initialse the collage.
ct = collage.CollageTiler(
    prompts=all_prompts,
    fixed_background_image=background_image,
    clip_model=clip_model,
    device=device,
    config=config)
ct.initialise()

# Collage optimisation loop.
output = ct.loop()

# Render high res image and finish up.
ct.assemble_tiles()

# Clean-up temporary files.
if config["clean_up"]:
  for file_match in ["*.npy", "tile_*.png"]:
    output_dir = config["output_dir"]
    files = glob.glob(f"{output_dir}/{file_match}")
    for f in files:
      os.remove(f)
