# This is a temporary (evil) local hack to move global settings from the main
# file in preparation for it to be replaced with a config file, e.g. yaml.
from arnheim_3_globals_config import *

# This is the beginning of a better way to specify (and override) the config.
import configargparse
#import configargparse
ap = configargparse.ArgumentParser()
# ap.add_argument("-c", "--config", is_config_file=True, help="Config file")
ap.add_argument('--cuda', dest='cuda', action='store_true')
ap.add_argument('--no-cuda', dest='cuda', action='store_false')
ap.set_defaults(cuda=True)
ap.add_argument("--output_dir", help="Output directory", default=None)
ap.add_argument("--clean_up", dest='clean_up', help="Remove all working files",
    action='store_true')
ap.add_argument("--no-clean_up", dest='clean_up', help="Remove all working files",
    action='store_false')
ap.set_defaults(clean_up=True)
# ap.add_argument('--gui', dest='gui', action='store_true')
# ap.add_argument('--no-gui', dest='gui', action='store_false')
# ap.set_defaults(gui=False)
# ap.add_argument("--num_augs", help="NUM_AUGS", default="4")
# ap.add_argument("--steps", help="OPTIM_STEPS", default=None)
# ap.add_argument("--num_patches", help="NUM_PATCHES", default="10")
args = vars(ap.parse_args())
print(ap.format_values())

# NUM_AUGS = int(args.get("augs"))
# OPTIM_STEPS = int(args.get("steps"))
# NUM_PATCHES = int(args.get("patches"))
_OUTPUT_DIR = args.get("output_dir")
if _OUTPUT_DIR is not None:
  OUTPUT_DIR = _OUTPUT_DIR
CLEAN_UP = args.get("clean_up")

# GUI = args.get("gui")

CUDA = args.get("cuda")
GUI = False
print(f"CUDA={CUDA}")

import subprocess

if CUDA:
  CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
  print("CUDA version:", CUDA_version)
  torch_device = "cuda"
else:
  torch_device = "cpu"

"""# Imports and libraries"""

#@title Imports {vertical-output: true}
import clip
import copy
import cv2
import glob
import io
from kornia.color import hsv
from matplotlib import pyplot as plt
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import numpy as np
import os
import pathlib
import random
import requests
from skimage.transform import resize
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


os.environ["FFMPEG_BINARY"] = "ffmpeg"
print("Torch version:", torch.__version__)

#@title Initialise and load CLIP model {vertical-output: true}

device = torch.device(torch_device)
CLIP_MODEL = "ViT-B/32"
print(f"Downloading CLIP model {CLIP_MODEL}...")
clip_model, _ = clip.load(CLIP_MODEL, device, jit=False)

CANVAS_WIDTH = 224
CANVAS_HEIGHT = 224
USE_EVOLUTION = POP_SIZE > 1

if COMPOSITIONAL_IMAGE:
  CANVAS_WIDTH *= 2
  CANVAS_HEIGHT *= 2
  MULTIPLIER_BIG_IMAGE = int(MULTIPLIER_BIG_IMAGE / 2)

if COMPOSITIONAL_IMAGE:
  print("Using ONE image augmentations for compositional image creation.")
  USE_IMAGE_AUGMENTATIONS = True
  NUM_AUGS = 1


# @title Saving images on Drive
#@markdown Displayed results can also be stored on Google Drive.
STORE_ON_GOOGLE_DRIVE = False  #@param {type:"boolean"}

DIR_RESULTS = f"{OUTPUT_DIR}"
print(f"Storing results in {DIR_RESULTS}")
pathlib.Path(DIR_RESULTS).mkdir(parents=True, exist_ok=True)

"""# Images patches"""

#@title Functions used for loading patches

def add_binary_alpha_mask(patch):
  """Black pixels treated as having alpha=0, all other pixels have alpha=255"""
  shape = patch.shape
  mask = ((patch.sum(2) > 0) * 255).astype(np.uint8)
  return np.concatenate([patch, np.expand_dims(mask, -1)], axis=-1)


def resize_patch(patch, coeff):
  return resize(patch.astype(float),
                (int(np.round(patch.shape[0] * coeff)),
                 int(np.round(patch.shape[1] * coeff))))


def print_size_segmented_data(segmented_data, show=True):
  size_max = 0
  shape_max = None
  size_min = np.infty
  shape_min = None
  ws = []
  hs = []
  for i, segment in enumerate(segmented_data):
    segment = segment.swapaxes(0, 1)
    shape_i = segment.shape
    size_i = shape_i[0] * shape_i[1]
    if size_i > size_max:
      shape_max = shape_i
      size_max = size_i
    if size_i < size_min:
      shape_min = shape_i
      size_min = size_i
    print(f'Patch {i} of shape {shape_i}')
    if show:
      im_i = cv2.cvtColor(segment, cv2.COLOR_RGBA2BGRA)
      im_bgr = im_i[:, :, :3]
      im_mask = np.tile(im_i[:, :, 3:], (1, 1, 3))
      im_render = np.concatenate([im_bgr, im_mask], 1)
      cv2_imshow(im_render)
  print(f"{len(segmented_data)} patches, max {shape_max}, min {shape_min}")


def cached_url_download(url):
  cache_filename = os.path.basename(url)
  cache = pathlib.Path(cache_filename)
  if not cache.is_file():
    print("Downloading " + cache_filename)
    r = requests.get(url)
    bytesio_object = io.BytesIO(r.content)
    with open(cache_filename, "wb") as f:
        f.write(bytesio_object.getbuffer())
  else:
    print("Using cached version of " + cache_filename)
  return np.load(cache, allow_pickle=True)

def upload_file():
  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    with open(fn, 'rb') as f:
      return np.load(f, allow_pickle = True)

examples = {"Fruit and veg" : "fruit.npy",
            "Sea glass" : "shore_glass.npy",
            #"Chrisantha" : "chrisantha2.npy",
            "Handwritten MNIST" : "handwritten_mnist.npy",
            "Animals" : "animals.npy"}

if PATCH_SET in examples:
  repo_root = "https://github.com/deepmind/arnheim/raw/main"
  segmented_data_initial = cached_url_download(
      f"{repo_root}/collage_patches/{examples[PATCH_SET]}")
elif PATCH_SET == "Load from URL":
  segmented_data_initial = cached_url_download(URL_TO_PATCH_FILE)
elif PATCH_SET == "Upload to Colab":
  segmented_data_initial = upload_file()
else:  # "Load from Google Drive"
  raise ValueError(f"Unsupported {PATCH_SET}")

segmented_data_initial_tmp = []
for i in range(len(segmented_data_initial)):
  if segmented_data_initial[i].shape[2] == 3:
    segmented_data_initial_tmp.append(add_binary_alpha_mask(
        segmented_data_initial[i]))
  else:
    segmented_data_initial_tmp.append(
        segmented_data_initial[i])

segmented_data_initial = segmented_data_initial_tmp

#@markdown Show all the patches (useful for dubugging)
SHOW_PATCHES = False #@param {type:"boolean"}

def normalise_patch_brightness(patch):
  max_intensity = max(patch.max(), 1.0)
  return ((patch / max_intensity) * 255).astype(np.uint8)

# Permute the order of the segmented images.
num_patches = len(segmented_data_initial)
order = np.random.permutation(num_patches)

# Compress all images until they are at most 1/PATCH_MAX_PROPORTION of the large
# canvas size.
hires_height = CANVAS_HEIGHT * MULTIPLIER_BIG_IMAGE
hires_width = CANVAS_WIDTH * MULTIPLIER_BIG_IMAGE
height_large_max = hires_height / PATCH_MAX_PROPORTION
width_large_max = hires_width / PATCH_MAX_PROPORTION
if FIXED_SCALE_PATCHES:
  print(f"Max size for fixed scale patches: ({hires_height},{hires_width})")
else:
  print(
      f"Max patch size on large image: ({height_large_max}, {width_large_max})")
segmented_data = []
segmented_data_high_res = []
for patch_i in range(num_patches):
  segmented_data_initial_i = segmented_data_initial[
      order[patch_i]].astype(np.float32).swapaxes(0, 1)
  shape_i = segmented_data_initial_i.shape
  h_i = shape_i[0]
  w_i = shape_i[1]
  if h_i >= PATCH_HEIGHT_MIN and w_i >= PATCH_WIDTH_MIN:
    # Coefficient for resizing the patch.
    if FIXED_SCALE_PATCHES:
      coeff_i_large = FIXED_SCALE_COEFF
      if h_i * coeff_i_large > hires_height:
        coeff_i_large = hires_height / h_i
      if w_i * coeff_i_large > width_large_max:
        coeff_i_large = min(coeff_i_large, hires_width / w_i)
      if coeff_i_large != FIXED_SCALE_COEFF:
        print(
            f"Patch {patch_i} too large; setting scaled to {coeff_i_large:.2f}")
    else:
      coeff_i_large = 1.0
      if h_i > height_large_max:
        coeff_i_large = height_large_max / h_i
      if w_i > width_large_max:
        coeff_i_large = min(coeff_i_large, width_large_max / w_i)

    # Resize the high-res patch?
    if coeff_i_large < 1.0:
      segmented_data_high_res_i = resize_patch(segmented_data_initial_i,
                                              coeff_i_large)
    else:
      segmented_data_high_res_i = np.copy(segmented_data_initial_i)

    # Resize the low-res patch.
    coeff_i = coeff_i_large / MULTIPLIER_BIG_IMAGE
    segmented_data_i = resize_patch(segmented_data_initial_i, coeff_i)
    if (segmented_data_i.shape[0] > CANVAS_HEIGHT
        or segmented_data_i.shape[1] > CANVAS_WIDTH):
      print(f"Patch size {segmented_data_i.shape} exceeds canvas size ({CANVAS_HEIGHT},{CANVAS_WIDTH})")
      import pdb; pdb.set_trace()
    if NORMALIZE_PATCH_BRIGHTNESS:
      segmented_data_i[...,:3] = normalise_patch_brightness(
          segmented_data_i[...,:3])
      segmented_data_high_res_i[...,:3] = normalise_patch_brightness(
          segmented_data_high_res_i[...,:3])
    segmented_data_high_res_i = segmented_data_high_res_i.astype(np.uint8)
    segmented_data_high_res.append(segmented_data_high_res_i)
    segmented_data_i = segmented_data_i.astype(np.uint8)
    segmented_data.append(segmented_data_i)
    #print("{}/{}: initial {} -> small {}, large {} x{:.2f}".format(
    #    patch_i, num_patches, shape_i, segmented_data_i.shape,
    #    segmented_data_high_res_i.shape,
    #    coeff_i_large))
  else:
    print(f"Discard patch of size {h_i}x{w_i}")

if SHOW_PATCHES:
  print("Patch sizes during optimisation:")
  print_size_segmented_data(segmented_data, show=GUI)
  print("Patch sizes for high-resolution final image:")
  print_size_segmented_data(segmented_data_high_res, show=GUI)

"""# Colour and affine transforms

## Affine transform classes
"""

class PopulationAffineTransforms(torch.nn.Module):
  """Population-based Affine Transform network."""
  def __init__(self, num_patches=1, pop_size=1):
    super(PopulationAffineTransforms, self).__init__()

    self._pop_size = pop_size
    matrices_translation = (
        np.random.rand(pop_size, num_patches, 2, 1) * (MAX_TRANS - MIN_TRANS)
        + MIN_TRANS)
    matrices_rotation = (
        np.random.rand(pop_size, num_patches, 1, 1) * (MAX_ROT - MIN_ROT)
        + MIN_ROT)
    matrices_scale = (
        np.random.rand(pop_size, num_patches, 1, 1) * (MAX_SCALE - MIN_SCALE)
        + MIN_SCALE)
    matrices_squeeze = (
        np.random.rand(pop_size, num_patches, 1, 1) * (
            (MAX_SQUEEZE - MIN_SQUEEZE) + MIN_SQUEEZE))
    matrices_shear = (
        np.random.rand(pop_size, num_patches, 1, 1) * (MAX_SHEAR - MIN_SHEAR)
        + MIN_SHEAR)
    self.translation = torch.nn.Parameter(
        torch.tensor(matrices_translation, dtype=torch.float),
        requires_grad=True)
    self.rotation = torch.nn.Parameter(
        torch.tensor(matrices_rotation, dtype=torch.float),
        requires_grad=True)
    self.scale = torch.nn.Parameter(
        torch.tensor(matrices_scale, dtype=torch.float),
        requires_grad=True)
    self.squeeze = torch.nn.Parameter(
        torch.tensor(matrices_squeeze, dtype=torch.float),
        requires_grad=True)
    self.shear = torch.nn.Parameter(
        torch.tensor(matrices_shear, dtype=torch.float),
        requires_grad=True)
    self._identity = (
        torch.ones((pop_size, num_patches, 1, 1)) * torch.eye(2).unsqueeze(0)
        ).to(device)
    self._zero_column = torch.zeros((pop_size, num_patches, 2, 1)).to(device)
    self._unit_row = (
        torch.ones((pop_size, num_patches, 1, 1)) * torch.tensor([0., 0., 1.])
        ).to(device)
    self._zeros = torch.zeros((pop_size, num_patches, 1, 1)).to(device)

  def _clamp(self):
    self.translation.data = self.translation.data.clamp(
        min=MIN_TRANS, max=MAX_TRANS)
    self.rotation.data = self.rotation.data.clamp(
        min=MIN_ROT, max=MAX_ROT)
    self.scale.data = self.scale.data.clamp(
        min=MIN_SCALE, max=MAX_SCALE)
    self.squeeze.data = self.squeeze.data.clamp(
        min=MIN_SQUEEZE, max=MAX_SQUEEZE)
    self.shear.data = self.shear.data.clamp(
        min=MIN_SHEAR, max=MAX_SHEAR)

  def copy_and_mutate_s(self, parent, child):
    """Copy parameters to child, mutating transform parameters."""
    with torch.no_grad():
      self.translation[child, ...] = (self.translation[parent, ...]
          + POS_AND_ROT_MUTATION_SCALE * torch.randn(
              self.translation[child, ...].shape).to(device))
      self.rotation[child, ...] = (self.rotation[parent, ...]
          + POS_AND_ROT_MUTATION_SCALE * torch.randn(
              self.rotation[child, ...].shape).to(device))
      self.scale[child, ...] = (self.scale[parent, ...]
          + SCALE_MUTATION_SCALE * torch.randn(
              self.scale[child, ...].shape).to(device))
      self.squeeze[child, ...] = (self.squeeze[parent, ...]
          + DISTORT_MUTATION_SCALE * torch.randn(
              self.squeeze[child, ...].shape).to(device))
      self.shear[child, ...] = (self.shear[parent, ...]
          + DISTORT_MUTATION_SCALE * torch.randn(
              self.shear[child, ...].shape).to(device))

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other spatial transform, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.translation[idx_to, ...] = other.translation[idx_from, ...]
      self.rotation[idx_to, ...] = other.rotation[idx_from, ...]
      self.scale[idx_to, ...] = other.scale[idx_from, ...]
      self.squeeze[idx_to, ...] = other.squeeze[idx_from, ...]
      self.shear[idx_to, ...] = other.shear[idx_from, ...]

  def forward(self, x):
    self._clamp()
    scale_affine_mat = torch.cat([
        torch.cat([self.scale, self.shear], 3),
        torch.cat([self._zeros, self.scale * self.squeeze], 3)],
        2)
    scale_affine_mat = torch.cat([
        torch.cat([scale_affine_mat, self._zero_column], 3),
        self._unit_row], 2)
    rotation_affine_mat = torch.cat([
        torch.cat([torch.cos(self.rotation), -torch.sin(self.rotation)], 3),
        torch.cat([torch.sin(self.rotation), torch.cos(self.rotation)], 3)],
        2)
    rotation_affine_mat = torch.cat([
        torch.cat([rotation_affine_mat, self._zero_column], 3),
        self._unit_row], 2)

    scale_rotation_mat = torch.matmul(scale_affine_mat,
                                      rotation_affine_mat)[:, :, :2, :]
    # Population and patch dimensions (0 and 1) need to be merged.
    # E.g. from (POP_SIZE, NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    # to (POP_SIZE * NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    scale_rotation_mat = scale_rotation_mat[:, :, :2, :].view(
        1, -1, *(scale_rotation_mat[:, :, :2, :].size()[2:])).squeeze()
    x = x.view(1, -1, *(x.size()[2:])).squeeze()
    scaled_rotated_grid = F.affine_grid(
        scale_rotation_mat, x.size(), align_corners=True)
    scaled_rotated_x = F.grid_sample(x, scaled_rotated_grid, align_corners=True)

    translation_affine_mat = torch.cat([self._identity, self.translation], 3)
    translation_affine_mat = translation_affine_mat.view(
        1, -1, *(translation_affine_mat.size()[2:])).squeeze()
    translated_grid = F.affine_grid(
        translation_affine_mat, x.size(), align_corners=True)
    y = F.grid_sample(scaled_rotated_x, translated_grid, align_corners=True)
    return y.view(self._pop_size, NUM_PATCHES, *(y.size()[1:]))

  def tensor_to(self, device):
    self.translation = self.translation.to(device)
    self.rotation = self.rotation.to(device)
    self.scale = self.scale.to(device)
    self.squeeze = self.squeeze.to(device)
    self.shear = self.shear.to(device)
    self._identity = self._identity.to(device)
    self._zero_column = self._zero_column.to(device)
    self._unit_row = self._unit_row.to(device)
    self._zeros = self._zeros.to(device)

"""## RGB and HSV color transforms"""

class PopulationOrderOnlyTransforms(torch.nn.Module):

  def __init__(self, num_patches=1, pop_size=1):
    super(PopulationOrderOnlyTransforms, self).__init__()

    self._pop_size = pop_size

    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=True)
    self._hsv_to_rgb = hsv.HsvToRgb()

  def _clamp(self):
    self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      self.orders[child, ...] = self.orders[parent, ...]

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other colour transform, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.orders[idx_to, ...] = other.orders[idx_from, ...]

  def forward(self, x):
    self._clamp()
    colours = torch.cat(
        [self._zeros, self._zeros, self._zeros, self._zeros, self.orders],
        2)
    return colours * x

  def tensor_to(self, device):
    self.orders = self.orders.to(device)
    self._zeros = self._zeros.to(device)


class PopulationColourHSVTransforms(torch.nn.Module):

  def __init__(self, num_patches=1, pop_size=1):
    super(PopulationColourHSVTransforms, self).__init__()

    print('PopulationColourHSVTransforms for {} patches, {} individuals'.format(
        num_patches, pop_size))
    self._pop_size = pop_size

    coeff_hue = 0.5 * (MAX_HUE - MIN_HUE) + MIN_HUE
    coeff_sat = 0.5 * (MAX_SAT - MIN_SAT) + MIN_SAT
    coeff_val = 0.5 * (MAX_VAL - MIN_VAL) + MIN_VAL
    population_hues = np.random.rand(pop_size, num_patches, 1, 1, 1) * coeff_hue
    population_saturations = np.random.rand(
        pop_size, num_patches, 1, 1, 1) * coeff_sat
    population_values = np.random.rand(
        pop_size, num_patches, 1, 1, 1) * coeff_val
    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self.hues = torch.nn.Parameter(
        torch.tensor(population_hues, dtype=torch.float),
        requires_grad=True)
    self.saturations = torch.nn.Parameter(
        torch.tensor(population_saturations, dtype=torch.float),
        requires_grad=True)
    self.values = torch.nn.Parameter(
        torch.tensor(population_values, dtype=torch.float),
        requires_grad=True)
    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=True)
    self._hsv_to_rgb = hsv.HsvToRgb()

  def _clamp(self):
    self.hues.data = self.hues.data.clamp(min=MIN_HUE, max=MAX_HUE)
    self.saturations.data = self.saturations.data.clamp(
        min=MIN_SAT, max=MAX_SAT)
    self.values.data = self.values.data.clamp(min=MIN_VAL, max=MAX_VAL)
    self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      self.hues[child, ...] = (
          self.hues[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.hues[child, ...].shape).to(device))
      self.saturations[child, ...] = (
          self.saturations[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.saturations[child, ...].shape).to(device))
      self.values[child, ...] = (
          self.values[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.values[child, ...].shape).to(device))
      self.orders[child, ...] = self.orders[parent, ...]

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other colour transform, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.hues[idx_to, ...] = other.hues[idx_from, ...]
      self.saturations[idx_to, ...] = other.saturations[idx_from, ...]
      self.values[idx_to, ...] = other.values[idx_from, ...]
      self.orders[idx_to, ...] = other.orders[idx_from, ...]

  def forward(self, image):
    self._clamp()
    colours = torch.cat(
        [self.hues, self.saturations, self.values, self._zeros, self.orders], 2)
    hsv_image = colours * image
    rgb_image = self._hsv_to_rgb(hsv_image[:, :, :3, :, :])
    return torch.cat([rgb_image, hsv_image[:, :, 3:, :, :]], axis=2)

  def tensor_to(self, device):
    self.hues = self.hues.to(device)
    self.saturations = self.saturations.to(device)
    self.values = self.values.to(device)
    self.orders = self.orders.to(device)
    self._zeros = self._zeros.to(device)


class PopulationColourRGBTransforms(torch.nn.Module):

  def __init__(self, num_patches=1, pop_size=1):
    super(PopulationColourRGBTransforms, self).__init__()

    print('PopulationColourRGBTransforms for {} patches, {} individuals'.format(
        num_patches, pop_size))
    self._pop_size = pop_size

    rgb_init_range = INITIAL_MAX_RGB - INITIAL_MIN_RGB
    population_reds = (np.random.rand(pop_size, num_patches, 1, 1, 1)
        * rgb_init_range) + INITIAL_MIN_RGB
    population_greens = (np.random.rand(
        pop_size, num_patches, 1, 1, 1) * rgb_init_range) + INITIAL_MIN_RGB
    population_blues = (np.random.rand(
        pop_size, num_patches, 1, 1, 1) * rgb_init_range) + INITIAL_MIN_RGB
    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self.reds = torch.nn.Parameter(
        torch.tensor(population_reds, dtype=torch.float),
        requires_grad=True)
    self.greens = torch.nn.Parameter(
        torch.tensor(population_greens, dtype=torch.float),
        requires_grad=True)
    self.blues = torch.nn.Parameter(
        torch.tensor(population_blues, dtype=torch.float),
        requires_grad=True)
    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=True)

  def _clamp(self):
    self.reds.data = self.reds.data.clamp(min=MIN_RGB, max=MAX_RGB)
    self.greens.data = self.greens.data.clamp(min=MIN_RGB, max=MAX_RGB)
    self.blues.data = self.blues.data.clamp(min=MIN_RGB, max=MAX_RGB)
    self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      self.reds[child, ...] = (
          self.reds[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.reds[child, ...].shape).to(device))
      self.greens[child, ...] = (
          self.greens[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.greens[child, ...].shape).to(device))
      self.blues[child, ...] = (
          self.blues[parent, ...]
          + COLOUR_MUTATION_SCALE * torch.randn(
              self.blues[child, ...].shape).to(device))
      self.orders[child, ...] = self.orders[parent, ...]

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other colour transform, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.reds[idx_to, ...] = other.reds[idx_from, ...]
      self.greens[idx_to, ...] = other.greens[idx_from, ...]
      self.blues[idx_to, ...] = other.blues[idx_from, ...]
      self.orders[idx_to, ...] = other.orders[idx_from, ...]

  def forward(self, x):
    self._clamp()
    colours = torch.cat(
        [self.reds, self.greens, self.blues, self._zeros, self.orders], 2)
    return colours * x

  def tensor_to(self, device):
    self.reds = self.reds.to(device)
    self.greens = self.greens.to(device)
    self.blues = self.blues.to(device)
    self.orders = self.orders.to(device)
    self._zeros = self._zeros.to(device)

"""# Rendering functions"""

RENDER_EPSILON = 1e-8
RENDER_OVERLAP_TEMPERATURE = 0.1
RENDER_OVERLAP_ZERO_OFFSET = -5
RENDER_OVERLAP_MASK_THRESHOLD = 0.5
RENDER_TRANSPARENCY_MASK_THRESHOLD = 0.1


def population_render_transparency(x, b=None):
  """Image rendering function that renders all patches on top of one another,
     with transparency, using black as the transparent colour.

  Args:
    x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
    b: optional tensor of background RGB image of shape [S, 3, H, W].
  Returns:
    Tensor of rendered RGB images of shape [S, 3, H, W].
  """
  # Sum the RGB patches [S, B, 3, H, W] as [S, 3, H, W].
  x = x[:, :, :3, :, :] * x[:, :, 3:4, :, :]
  y = x[:, :, :3, :, :].sum(1)
  if INVERT_COLOURS:
    y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
  # Add backgrounds [S, 3, H, W].
  if b is not None:
    b = b.cuda() if x.is_cuda else b.cpu()
    y = torch.where(y.sum(1, keepdim=True) > RENDER_TRANSPARENCY_MASK_THRESHOLD,
                    y[:, :3, :, :], b.unsqueeze(0)[:, :3, :, :])
  return y.clamp(0., 1.).permute(0, 2, 3, 1)


def population_render_masked_transparency(x, b=None):
  """Image rendering function that renders all patches on top of one another,
     with transparency, using the alpha chanel as the mask colour.

  Args:
    x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
    b: optional tensor of background RGB image of shape [S, 3, H, W].
  Returns:
    Tensor of rendered RGB images of shape [S, 3, H, W].
  """
  # Get the patch mask [S, B, 1, H, W] and sum of masks [S, 1, H, W].
  mask = x[:, :, 3:4, :, :]
  mask_sum = mask.sum(1) + RENDER_EPSILON
  # Mask the RGB patches [S, B, 4, H, W] -> [S, B, 3, H, W].
  masked_x = x[:, :, :3, :, :] * mask
  # Compute mean of the RGB patches [S, B, 3, H, W] as [S, 3, H, W].
  x_sum = masked_x.sum(1)
  y = torch.where(
      mask_sum > RENDER_EPSILON, x_sum / mask_sum, mask_sum)
  # Anti-aliasing on the countours of the sum of patches.
  y = y * mask_sum.clamp(0., 1.)
  if INVERT_COLOURS:
    y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
  # Add backgrounds [S, 3, H, W].
  if b is not None:
    b = b.cuda() if x.is_cuda else b.cpu()
    y = torch.where(mask.sum(1) > RENDER_OVERLAP_MASK_THRESHOLD, y[:, :3, :, :],
                  b.unsqueeze(0)[:, :3, :, :])
  return y.clamp(0., 1.).permute(0, 2, 3, 1)


def population_render_overlap(x, b=None, gamma=None):
  """Image rendering function that overlays all patches on top of one another,
     with semi-translucent overlap, using the alpha chanel as the mask colour
     and the 5th channel as the order for the overlapped images.

  Args:
    x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
    b: optional tensor of background RGB image of shape [S, 3, H, W].
  Returns:
    Tensor of rendered RGB images of shape [S, 3, H, W].
  """
  # Get the patch mask [S, B, 1, H, W].
  mask = x[:, :, 3:4, :, :]
  # Mask the patches [S, B, 4, H, W] -> [S, B, 3, H, W]
  masked_x = x[:, :, :3, :, :] * mask * mask
  # Mask the orders [S, B, 1, H, W] -> [S, B, 1, H, W]
  order = torch.where(
      mask > RENDER_OVERLAP_MASK_THRESHOLD,
      x[:, :, 4:, :, :] * mask / RENDER_OVERLAP_TEMPERATURE,
      mask + RENDER_OVERLAP_ZERO_OFFSET)
  # Get weights from orders [S, B, 1, H, W]
  weights = F.softmax(order, dim=1)
  # Apply weights to masked patches and compute mean over patches [S, 3, H, W].
  y = (weights * masked_x).sum(1)
  if INVERT_COLOURS:
    y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
  if b is not None:
    b = b.cuda() if x.is_cuda else b.cpu()
    y = torch.where(mask.sum(1) > RENDER_OVERLAP_MASK_THRESHOLD, y[:, :3, :, :],
                  b.unsqueeze(0)[:, :3, :, :])
  return y.clamp(0., 1.).permute(0, 2, 3, 1)

"""# Collage network definition"""

class PopulationCollage(torch.nn.Module):
  """Population-based segmentation collage network.

  Image structure in this class is SCHW."""
  def __init__(self,
               pop_size=1,
               is_high_res=False,
               segmented_data=None,
               background_image=None):
    """Constructor, relying on global parameters."""
    super(PopulationCollage, self).__init__()

    # Population size.
    self._pop_size = pop_size

    # Create the spatial transformer and colour transformer for patches.
    if CUDA:
      self.spatial_transformer = PopulationAffineTransforms(
          num_patches=NUM_PATCHES, pop_size=pop_size).cuda()
      if COLOUR_TRANSFORMATIONS == "HSV space":
        self.colour_transformer = PopulationColourHSVTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size).cuda()
      elif COLOUR_TRANSFORMATIONS == "RGB space":
        self.colour_transformer = PopulationColourRGBTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size).cuda()
      else:
        self.colour_transformer = PopulationOrderOnlyTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size).cuda()
      self.spatial_transformer = PopulationAffineTransforms(
          num_patches=NUM_PATCHES, pop_size=pop_size).cuda()
    else:
      self.spatial_transformer = PopulationAffineTransforms(
          num_patches=NUM_PATCHES, pop_size=pop_size)
      if COLOUR_TRANSFORMATIONS == "HSV space":
        self.colour_transformer = PopulationColourHSVTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size)
      elif COLOUR_TRANSFORMATIONS == "RGB space":
        self.colour_transformer = PopulationColourRGBTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size)
      else:
        self.colour_transformer = PopulationOrderOnlyTransforms(
            num_patches=NUM_PATCHES, pop_size=pop_size)

    # Optimisation is run in low-res, final rendering is in high-res.
    self._high_res = is_high_res

    # Store the background image (low- and high-res).
    self._background_image = background_image
    if self._background_image is not None:
      print(f'Background image of size {self._background_image.shape}')

    # Store the dataset (low- and high-res).
    self._dataset = segmented_data
    #print(f'There are {len(self._dataset)} image patches in the dataset')

    # Initial set of indices, pointing to the NUM_PATCHES first dataset images.
    self.patch_indices = [np.arange(NUM_PATCHES) % len(self._dataset)
                          for _ in range(pop_size)]

    # Patches in low and high-res.
    self.patches = None
    self.store_patches()

  def store_patches(self, population_idx=None):
    """Store the image patches for each population element."""
    t0 = time.time()

    if population_idx is not None and self.patches is not None:
      list_indices = [population_idx]
      #print(f'Reload {NUM_PATCHES} image patches for [{population_idx}]')
      self.patches[population_idx, :, :4, :, :] = 0
    else:
      list_indices = np.arange(self._pop_size)
      #print(f'Store {NUM_PATCHES} image patches for [1, ..., {self._pop_size}]')
      if self._high_res:
        self.patches = torch.zeros(
            1, NUM_PATCHES, 5, CANVAS_HEIGHT * MULTIPLIER_BIG_IMAGE,
            CANVAS_WIDTH * MULTIPLIER_BIG_IMAGE).to('cpu')
      else:
        self.patches = torch.zeros(
            self._pop_size, NUM_PATCHES, 5, CANVAS_HEIGHT, CANVAS_WIDTH
            ).to(device)
      self.patches[:, :, 4, :, :] = 1.0

    # Put the segmented data into the patches.
    for i in list_indices:
      for j in range(NUM_PATCHES):
        k = self.patch_indices[i][j]
        patch_j = torch.tensor(
            self._dataset[k].swapaxes(0, 2) / 255.0).to(device)
        width_j = patch_j.shape[1]
        height_j = patch_j.shape[2]
        if self._high_res:
          w0 = int((CANVAS_WIDTH * MULTIPLIER_BIG_IMAGE - width_j) / 2.0)
          h0 = int((CANVAS_HEIGHT * MULTIPLIER_BIG_IMAGE - height_j) / 2.0)
        else:
          w0 = int((CANVAS_WIDTH - width_j) / 2.0)
          h0 = int((CANVAS_HEIGHT - height_j) / 2.0)
        if w0 < 0 or h0 < 0:
          import pdb; pdb.set_trace()
        self.patches[i, j, :4, w0:(w0 + width_j), h0:(h0 + height_j)] = patch_j
    t1 = time.time()
    #print('Updated patches in {:.3f}s'.format(t1-t0))

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      # Copy the patches indices from the parent to the child.
      self.patch_indices[child] = copy.deepcopy(self.patch_indices[parent])

      # Mutate the child patches with a single swap from the original dataset.
      if PATCH_MUTATION_PROBABILITY > np.random.uniform():
        idx_dataset  = np.random.randint(len(self._dataset))
        idx_patch  = np.random.randint(NUM_PATCHES)
        self.patch_indices[child][idx_patch] = idx_dataset

      # Update all the patches for the child.
      self.store_patches(child)

      self.spatial_transformer.copy_and_mutate_s(parent, child)
      self.colour_transformer.copy_and_mutate_s(parent, child)

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other collage generator, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.patch_indices[idx_to] = copy.deepcopy(other.patch_indices[idx_from])
      self.store_patches(idx_to)
      self.spatial_transformer.copy_from(
          other.spatial_transformer, idx_to, idx_from)
      self.colour_transformer.copy_from(
          other.colour_transformer, idx_to, idx_from)

  def forward(self, params=None):
    """Input-less forward function."""

    shifted_patches = self.spatial_transformer(self.patches)
    background_image = self._background_image
    coloured_patches = self.colour_transformer(shifted_patches)
    if RENDER_METHOD == "transparency":
      img = population_render_transparency(coloured_patches, background_image)
    elif RENDER_METHOD == "masked_transparency":
      img = population_render_masked_transparency(
          coloured_patches, background_image)
    elif RENDER_METHOD == "opacity":
      if params is not None and 'gamma' in params:
        gamma = params['gamma']
      else:
        gamma = None
      img = population_render_overlap(
          coloured_patches, background_image)
    else:
      print("Unhandled render method")
    return img

  def tensors_to(self, device):
    self.spatial_transformer.tensor_to(device)
    self.colour_transformer.tensor_to(device)
    self.patches = self.patches.to(device)

"""# Image and video function definitions"""

#@title Image rendering and display

def layout_img_batch(img_batch, max_display=None):
    # img_batch.shape = (7, 224, 224, 3)  S, H, W, C
    img_np = img_batch.transpose(0, 2, 1, 3).clip(0.0, 1.0)  # S, W, H, C
    if max_display:
      img_np = img_np[:max_display, ...]
    sp = img_np.shape
    img_np[:, 0, :, :] = 1.0  # White line separator
    img_stitch = np.reshape(img_np, (sp[1] * sp[0], sp[2], sp[3]))
    img_r = img_stitch.transpose(1, 0, 2)   # H, W, C
    return img_r

def show_and_save(img_batch, t=None,
                  max_display=1, interpolation="None", stitch=True,
                  img_format="SCHW", show=True, filename=None):
  """Display image.

  Args:

    img: image to display
    t: time step
    max_display: max number of images to display from population
    interpolation: interpolate enlarged images
    stitch: append images side-by-side
    img_format: SHWC or SCHW (the latter used by CLIP)

  Returns:
    stitched image or None
  """

  if isinstance(img_batch, torch.Tensor):
    img_np = img_batch.detach().cpu().numpy()
  else:
    img_np = img_batch

  if len(img_np.shape) == 3:
    # if not a batch make it one.
    img_np = np.expand_dims(img_np, axis=0)

  if not stitch:
    # print(f"image (not stitch) min {img_np.min()}, max {img_np.max()}")
    for i in range(min(max_display, img_np.shape[0])):
      img = img_np[i]
      if img_format == "SCHW":  # Convert to SHWC
        img = np.transpose(img, (1, 2, 0))
      img = np.clip(img, 0.0, 1.0)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255
      if filename is not None:
        if img.shape[1] > CANVAS_WIDTH:
          filename = "highres_" + filename
        filename = f"{DIR_RESULTS}/{filename}_{str(i)}"
        if t is not None:
          filename += "_t_" + str(t)
        filename += ".png"
        print(f"Saving image {filename} (shape={img.shape})")
        cv2.imwrite(filename, img)
      if show:
        cv2_imshow(img)
    return None
  else:
    # print(f"image (stitch) min {img_np.min()}, max {img_np.max()}")
    img_np = np.clip(img_np, 0.0, 1.0)
    num_images = img_np.shape[0]
    if img_format == "SCHW":  # Convert to SHWC
      img_np = img_np.transpose((0, 2, 3, 1))
    laid_out = layout_img_batch(img_np, max_display)
    if show:
      cv2_imshow(cv2.cvtColor(laid_out, cv2.COLOR_BGR2RGB) * 255)
    return laid_out

#@title Video creator {vertical-output: true}

class VideoWriter:
  """Create a video from image frames."""

  def __init__(self, filename="_autoplay.mp4", fps=20.0, **kw):
    """Video creator.

    Creates and display a video made from frames. The default
    filename causes the video to be displayed on exit.

    Args:
      filename: name of video file
      fps: frames per second for video
      **kw: args to be passed to FFMPEG_VideoWriter

    Returns:
      VideoWriter instance.
    """

    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)
    print("No video writing implemented")

  def add(self, img):
    """Add image to video.

    Add new frame to image file, creating VideoWriter if requried.

    Args:
      img: array-like frame, shape [X, Y, 3] or [X, Y]

    Returns:
      None
    """
    pass
    # img = np.asarray(img)
    # if self.writer is None:
    #   h, w = img.shape[:2]
    #   self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    # if img.dtype in [np.float32, np.float64]:
    #   img = np.uint8(img.clip(0, 1)*255)
    # if len(img.shape) == 2:
    #   img = np.repeat(img[..., None], 3, -1)
    # self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params["filename"] == "_autoplay.mp4":
      self.show()

  def show(self, **kw):
    """Display video.

    Args:
      **kw: args to be passed to mvp.ipython_display

    Returns:
      None
    """
    self.close()
    fn = self.params["filename"]
    if GUI:
      display(mvp.ipython_display(fn, **kw))

#@title Metadata export

def export_metadata(metadata_filename):
  metadata = {}
  metadata_canvas = {"CANVAS_WIDTH": CANVAS_WIDTH,
                    "CANVAS_HEIGHT": CANVAS_HEIGHT,
                    "MULTIPLIER_BIG_IMAGE": MULTIPLIER_BIG_IMAGE,
                    "PATCH_WIDTH_MIN": PATCH_WIDTH_MIN,
                    "PATCH_HEIGHT_MIN": PATCH_HEIGHT_MIN,
                    "PATCH_MAX_PROPORTION": PATCH_MAX_PROPORTION}
  metadata = {"canvas": metadata_canvas}
  metadata_collage = {"RENDER_METHOD": RENDER_METHOD,
                      "NUM_PATCHES": NUM_PATCHES,
                      "COLOUR_TRANSFORMATIONS": COLOUR_TRANSFORMATIONS,
                      "INVERT_COLOURS": INVERT_COLOURS}
  metadata["collage"] = metadata_collage
  metadata_affine = {"MIN_TRANS": MIN_TRANS,
                    "MAX_TRANS": MAX_TRANS,
                    "MIN_SCALE": MIN_SCALE,
                    "MAX_SCALE": MAX_SCALE,
                    "MIN_SQUEEZE": MIN_SQUEEZE,
                    "MAX_SQUEEZE": MAX_SQUEEZE,
                    "MIN_SHEAR": MIN_SHEAR,
                    "MAX_SHEAR": MAX_SHEAR,
                    "MIN_ROT_DEG": MIN_ROT_DEG,
                    "MAX_ROT_DEG": MAX_ROT_DEG,
                    "MIN_ROT": MIN_ROT,
                    "MAX_ROT": MAX_ROT}
  metadata["affine"] = metadata_affine
  metadata_colour = {"MIN_RGB": MIN_RGB,
                    "MAX_RGB": MAX_RGB,
                    "MIN_HUE": MIN_HUE,
                    "MAX_HUE_DEG": MAX_HUE_DEG,
                    "MAX_HUE": MAX_HUE,
                    "MIN_SAT": MIN_SAT,
                    "MAX_SAT": MAX_SAT,
                    "MIN_VAL": MIN_VAL,
                    "MAX_VAL": MAX_VAL}
  metadata["colour"] = metadata_colour
  metadata_training = {"OPTIM_STEPS": OPTIM_STEPS,
                      "LEARNING_RATE": LEARNING_RATE,
                      "USE_IMAGE_AUGMENTATIONS": USE_IMAGE_AUGMENTATIONS,
                      "NUM_AUGS": NUM_AUGS,
                      "USE_NORMALIZED_CLIP": USE_NORMALIZED_CLIP,
                      "GRADIENT_CLIPPING": GRADIENT_CLIPPING,
                      "INITIAL_SEARCH_SIZE": INITIAL_SEARCH_SIZE}
  metadata["training"] = metadata_training
  metadata_evolution = {"POP_SIZE": POP_SIZE,
                        "EVOLUTION_FREQUENCY": EVOLUTION_FREQUENCY,
                        "GA_METHOD": GA_METHOD,
                        "POS_AND_ROT_MUTATION_SCALE": POS_AND_ROT_MUTATION_SCALE,
                        "SCALE_MUTATION_SCALE": SCALE_MUTATION_SCALE,
                        "DISTORT_MUTATION_SCALE": DISTORT_MUTATION_SCALE,
                        "COLOUR_MUTATION_SCALE": COLOUR_MUTATION_SCALE,
                        "PATCH_MUTATION_PROBABILITY": PATCH_MUTATION_PROBABILITY,
                        "MAX_MULTIPLE_VISUALISATIONS": MAX_MULTIPLE_VISUALISATIONS,
                        "USE_EVOLUTION": USE_EVOLUTION}
  metadata["evolution"] = metadata_evolution
  metadata_patches = {"PATCH_SET": PATCH_SET,
                      "URL_TO_PATCH_FILE": URL_TO_PATCH_FILE,
                      "DRIVE_PATH_TO_PATCH_FILE": DRIVE_PATH_TO_PATCH_FILE,
                      "NORMALIZE_PATCH_BRIGHTNESS": NORMALIZE_PATCH_BRIGHTNESS,
                      "FIXED_SCALE_PATCHES": FIXED_SCALE_PATCHES,
                      "FIXED_SCALE_COEFF": FIXED_SCALE_COEFF}
  metadata["patches"] = metadata_patches

  metadata_prompt = {"PROMPT": PROMPT,
                    "VIDEO_STEPS": VIDEO_STEPS,
                    "TRACE_EVERY": TRACE_EVERY}
  metadata["prompt"] = metadata_prompt

  # Write metadata to a Python-interpretable text file.
  with open(metadata_filename, "w") as f:
    for config_key, config in metadata.items():
      f.write(f"# {config_key}\n")
      for key, value in config.items():
        if isinstance(value, str):
          f.write(f"{key} = \"{value}\"\n")
        else:
          f.write(f"{key} = {value}\n")

"""# Training and evolution function definitions"""

#@title Image augmentation transformations for training

def augmentation_transforms(canvas_width,
                            use_normalized_clip=False,
                            use_augmentation=False):
  """Image transforms to produce distorted crops to augment the evaluation.

  Args:
    canvas_width: width of the drawing canvas
    use_normalized_clip: Normalisation to better suit CLIP's training data
    use_augmentation: Image augmentation by affine transform

  Returns:
    transforms
  """
  if use_normalized_clip and use_augmentation:
    augment_trans = transforms.Compose(
        [transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
         transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
         transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))])
  elif use_augmentation:
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
        transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
    ])
  elif use_normalized_clip:
    augment_trans = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
  else:
    augment_trans = transforms.RandomPerspective(
        fill=1, p=0, distortion_scale=0)

  return augment_trans

#@title Training functions

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_and_save_losses(loss_history, title="Losses", filename=None):
  losses = np.array(loss_history)
  plt.figure(figsize=(10,10))
  plt.xlabel("Training steps")
  plt.ylabel("Loss")
  plt.title(title)
  plt.plot(moving_average(losses, n=3))
  if filename:
    np.save(filename + ".npy", losses, allow_pickle=True)
    plt.savefig(filename + ".png")
  plt.show()


def make_optimizer(generator, learning_rate):
  """Make optimizer for generator's parameters.

  Args:
    generator: generator model
    learning_rate: learning rate
    input_learing_rate: learning rate for input

  Returns:
    optimizer
  """

  my_list = ['positions_top']
  params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list,
                                               generator.named_parameters()))))
  base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in
                                      my_list, generator.named_parameters()))))
  lr_scheduler = torch.optim.SGD([{'params': base_params},
                                  {'params': params, 'lr': learning_rate}],
                                    lr=learning_rate)
  return lr_scheduler


def text_features(prompts):
  # Compute CLIP features for all prompts.
  text_inputs = []
  for prompt in prompts:
    text_inputs.append(clip.tokenize(prompt).to(device))

  features = []
  with torch.no_grad():
    for text_input in text_inputs:
      features.append(clip_model.encode_text(text_input))
  return features


def create_augmented_batch(images, augment_trans, text_features):
  """Create batch of images to be evaluated.

  Returns:
    img_batch: For compositional images the batch contains all the regions.
        Otherwise the batch contains augmented versions of the original images.
    num_augs: number of images per original image
    expanded_text_features: a text feature for each augmentation
    loss_weights: weights for loss associated with each augmentation image
  """
  images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
  expanded_text_features = []
  if USE_IMAGE_AUGMENTATIONS:
    num_augs = NUM_AUGS
    img_augs = []
    for n in range(NUM_AUGS):
      img_n = augment_trans(images)
      img_augs.append(img_n)
      expanded_text_features.append(text_features[0])
    img_batch = torch.cat(img_augs)
    # Given images [P0, P1] and augmentations [a0(), a1()], output format:
    # [a0(P0), a0(P1), a1(P0), a1(P1)]
  else:
    num_augs = 1
    img_batch = augment_trans(images)
    expanded_text_features.append(text_features[0])
  return img_batch, num_augs, expanded_text_features, [1] * NUM_AUGS


def create_compositional_batch(images, augment_trans, text_features):
  """Create 10 sub-images per image by augmenting each with 3x3 crops.

  Args:
    images: population of N images, format [N, C, H, W]

  Returns:
    Tensor of all compositional sub-images + originals; [N*10, C, H, W] format:
        [x0_y0(P0) ... x0_y0(PN), ..., x2_y2(P0) ... x2_y2(PN), P0, ..., PN]
    10: Number of sub-images + whole, per original image.
    expanded_text_features: list of text features, 1 for each composition image
    loss_weights: weights for the losses corresponding to each composition image
    """
  if len(text_features) != 10:
    # text_features should already be 10 in size.
    raise ValueError(
        "10 text prompts required for compositional image creation")
  resize_for_clip = transforms.Compose([transforms.Scale((224,224))])
  img_swap = torch.swapaxes(images, 3, 1)
  ims = []
  i = 0
  for x in range(3):
    for y in range(3):
      #  print(f"Prompt for x={x}, y={y} is {PROMPTS[i]}")
      for k in range(images.shape[0]):
        ims.append(resize_for_clip(
            img_swap[k][:, y * 112 : y * 112 + 224, x * 112 : x * 112 + 224]))
      i += 1

  # Top-level (whole) images
  for k in range(images.shape[0]):
    ims.append(resize_for_clip(img_swap[k]))
  all_img = torch.stack(ims)
  all_img = torch.swapaxes(all_img, 1, 3)
  all_img = all_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
  all_img = augment_trans(all_img)

  # Last image gets 9 times as much weight
  common_weight = 1 / 5
  loss_weights = [common_weight] * 9
  loss_weights.append(9 * common_weight)
  return all_img, 10, text_features, loss_weights

#@title Evaluation and step of optimization

# Show each image being evaluated for debugging purposes.
VISUALISE_BATCH_IMAGES = False

def evaluation(t, clip_enc, generator, augment_trans, text_features,
               compositional_image, prompts):
  """Do a step of evaluation, returning images and losses.

  Args:
    t: step count
    clip_enc: model for CLIP encoding
    generator: drawing generator to optimise
    augment_trans: transforms for image augmentation
    text_features: tuple with the prompt two negative prompts
    compositional_image: use multiple CLIPs and prompts
    prompts: for debugging/visualisation - the list of text prompts

  Returns:
    loss: torch.Tensor of single combines loss
    losses_separate_np: numpy array of loss for each image
    losses_individuals_np: numpy array with loss for each population individual
    img_np: numpy array of images from the generator
  """

  # Annealing parameters.
  params = {'gamma': t / OPTIM_STEPS}

  # Rebuild the generator.
  img = generator(params)
  img_np = img.detach().cpu().numpy()

  # Create images for different regions
  pop_size = img.shape[0]
  if compositional_image:
    (img_batch, num_augs, text_features, loss_weights
     ) = create_compositional_batch(img, augment_trans, text_features)
  else:
    (img_batch, num_augs, text_features, loss_weights
     ) = create_augmented_batch(img, augment_trans, text_features)
  losses = torch.zeros(pop_size, num_augs).to(device)

  # Compute and add losses after augmenting the image with transforms.
  img_batch = torch.clip(img_batch, 0, 1)  # clip the images.
  image_features = clip_enc.encode_image(img_batch)
  count = 0
  for n in range(num_augs):  # number of augmentations or composition images
    for p in range(pop_size):
      loss = torch.cosine_similarity(
          text_features[n], image_features[count:count+1], dim=1
          )[0] * loss_weights[n]
      losses[p, n] -= loss
      if VISUALISE_BATCH_IMAGES and t % 500 == 0:
        # Show all the images in the batch along with their losses.
        if compositional_image:
          print(f"Loss {loss} for image region with prompt {prompts[n]}:")
        else:
          print(f"Loss {loss} for image augmentation with prompt {prompts[0]}:")
        show_and_save(img_batch[count].unsqueeze(0), img_format="SCHW",
            show=GUI)
      count += 1
  loss = torch.sum(losses) / pop_size
  losses_separate_np = losses.detach().cpu().numpy()
  # Sum losses for all each population individual.
  losses_individuals_np = losses_separate_np.sum(axis=1)
  return loss, losses_separate_np, losses_individuals_np, img_np


def step_optimization(t, clip_enc, lr_scheduler, generator, augment_trans,
                      text_features, compositional_image, prompts, output_dir,
                      final_step=False):
  """Do a step of optimization.

  Args:
    t: step count
    clip_enc: model for CLIP encoding
    lr_scheduler: optimizer
    generator: drawing generator to optimise
    augment_trans: transforms for image augmentation
    text_features: list or 1 or 9 prompts for normal and compositional creation
    compositional_image: use multiple CLIPs and prompts
    prompts: for debugging/visualisation - the list of text prompts
    final_step: if True does extras such as saving the model

  Returns:
    losses_np: numpy array with loss for each population individual
    losses_separate_np: numpy array of loss for each image
  """

  # Anneal learning rate and other parameters.
  if t == int(OPTIM_STEPS / 3):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0
  if t == int(OPTIM_STEPS * (2/3)):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0
  params = {'gamma': t / OPTIM_STEPS}

  # Forward pass.
  lr_scheduler.zero_grad()
  loss, losses_separate_np, losses_np, img_np = evaluation(
      t, clip_enc, generator, augment_trans, text_features, compositional_image,
      prompts)

  # Backpropagate the gradients.
  loss.backward()
  torch.nn.utils.clip_grad_norm(generator.parameters(), GRADIENT_CLIPPING)

  # Decay the learning rate.
  lr_scheduler.step()

  # Render the big version.
  if final_step:
    show_and_save(img_np, t=t, img_format="SHWC", show=GUI)
    print("Saving model...")
    torch.save(generator.state_dict(), f"{output_dir}/generator.pt")

  if t % TRACE_EVERY == 0:
    show_and_save(img_np,
                  max_display=MAX_MULTIPLE_VISUALISATIONS,
                  stitch=True, img_format="SHWC", show=GUI)

    print("Iteration {:3d}, rendering loss {:.6f}".format(t, loss.item()))
  return losses_np, losses_separate_np, img_np

#@title Evolution functions

def population_evolution_step(generator, losses):
  """GA for the population."""
  if GA_METHOD == "Microbial":
    # Competition between 2 random individuals; mutated winner replaces loser.
    indices = list(range(len(losses)))
    random.shuffle(indices)
    select_1, select_2 = indices[0], indices[1]
    if losses[select_1] < losses[select_2]:
      # print(f"Replacing {select_2} with {select_1}")
      generator.copy_and_mutate_s(select_1, select_2)
    else:
      # print(f"Replacing {select_1} with {select_2}")
      generator.copy_and_mutate_s(select_2, select_1)
  elif GA_METHOD == "Evolutionary Strategies":
    # Replace rest of population with mutants of the best."""
    winner = np.argmin(losses)
    for other in range(len(losses)):
      if other == winner:
        continue
      generator.copy_and_mutate_s(winner, other)

"""# Collage-making class definitions"""

#@title CollageMaker class

class CollageMaker():
  def __init__(
      self,
      prompts,
      segmented_data,
      background_image,
      compositional_image,
      output_dir,
      file_basename,
      video_steps,
      population_video,
      use_normalised_clip,
      use_image_augmentations,
      use_evolution,
      evolution_frequency,
      population_size,
      initial_search_size,
      optimisation_steps,
      ):
    """Create a single square collage image.

    Args:
      prompts: list of prompts. Optional compositional prompts plus a global one
      segmented_data: patches for the collage
      background_image: background image for the collage
      compositional_image: bool, whether to use 3x3 CLIPs
      output_dir: string, directory to save working and final images
      file_basename: string, name to use for the saved files
      video_steps: int, how many steps between video frames
      population_video: bool, create a video with members of the population
      use_normalised_clip: bool, colour-correct images for CLIP evaluation
      use_image_augmentations: bool, produce image augmentations for evaluation
      use_evolution: bool, turn evolution on/off
      evolution_frequency: bool, how many steps between evolution evaluations
      population_size: int, size of population being evolvec
      initial_search_size: int, samples evaluated for each intial network
      optimisation_steps: int, training steps for the collage
    """
    self._prompts = prompts
    self._segmented_data = segmented_data
    self._background_image = background_image
    self._compositional_image = compositional_image
    self._file_basename = file_basename
    self._output_dir = output_dir
    self._optimisation_steps = optimisation_steps
    self._population_video = population_video
    self._use_evolution = use_evolution
    self._evolution_frequency = evolution_frequency

    self._video_steps = video_steps
    if self._video_steps:
      self._video_writer = VideoWriter(
          filename=f"{self._output_dir}/{self._file_basename}.mp4")
      if self._population_video:
        self._population_video_writer = VideoWriter(
            filename=f"{self._output_dir}/{self._file_basename}_pop_sample.mp4")

    if self._compositional_image:
      if len(self._prompts) != 10:
        raise ValueError(
            "Missing compositional image prompts; found {len(self._prompts)}")
      print("Global prompt is", self._prompts[-1])
      print("Composition prompts", self._prompts)
    else:
      if len(self._prompts) != 1:
        raise ValueError(
            "Missing compositional image prompts; found {len(self._prompts)}")
      print("CLIP prompt", self._prompts[0])

    # Prompt to CLIP features.
    self._prompt_features = text_features(self._prompts)
    self._augmentations = augmentation_transforms(
        224,
        use_normalized_clip=use_normalised_clip,
        use_augmentation=use_image_augmentations)

    # Create population of collage generators.
    self._generator = PopulationCollage(
        is_high_res=False,
        pop_size=population_size,
        segmented_data=segmented_data,
        background_image=background_image)

    # Initial search over hyper-parameters.
    if initial_search_size > 1:
      print(f'\nInitial random search over {INITIAL_SEARCH_SIZE} individuals')
      for j in range(population_size):
        generator_search = PopulationCollage(
            is_high_res=False,
            pop_size=initial_search_size,
            segmented_data=segmented_data,
            background_image=background_image,
            compositional_image=self._compositional_image)
        _, _, losses, _ = evaluation(
            0, clip_model, generator_search, augmentations, prompt_features)
        print(f"Search {losses}")
        idx_best = np.argmin(losses)
        generator.copy_from(generator_search, j, idx_best)
        del generator_search
      print(f'Initial random search done\n')

    self._optimizer = make_optimizer(self._generator, LEARNING_RATE)
    self._step = 0
    self._losses_history = []
    self._losses_separated_history = []

  @property
  def generator(self):
    return self._generator

  @property
  def step(self):
    return self._step

  def loop(self):
    """Main optimisation/image generation loop. Can be interrupted."""
    if self._step == 0:
      print('Starting optimization of collage.')
    else:
      print(f'Continuing optimization of collage at step {self._step}.')
      if self._video_steps:
        print(f"Aborting video creation (does not work when interrupted).")
        self._video_steps = 0
        self._video_writer = None
        if self._population_video_writer:
          self._population_video_writer = None

    while self._step < self._optimisation_steps:
      last_step = self._step == (self._optimisation_steps - 1)
      losses, losses_separated, img_batch = step_optimization(
          self._step, clip_model, self._optimizer, self._generator,
          self._augmentations, self._prompt_features,
          self._compositional_image, self._prompts, self._output_dir,
          final_step=last_step)
      self._add_video_frames(img_batch, losses)
      self._losses_history.append(losses)
      self._losses_separated_history.append(losses_separated)

      if (self._use_evolution and self._step
          and self._step % self._evolution_frequency == 0):
        population_evolution_step(self._generator, losses)
      self._step += 1


  def high_res_render(self,
                      segmented_data_high_res,
                      background_image_high_res,
                      gamma=1.0,
                      show=True,
                      save=True):
    """Save and/or show a high res render using high-res patches."""
    generator = PopulationCollage(
        is_high_res=True,
        pop_size=1,
        segmented_data=segmented_data_high_res,
        background_image=background_image_high_res)
    idx_best = np.argmin(self._losses_history[-1])
    print(f'Lowest loss for indices: {idx_best}')
    generator.copy_from(self._generator, 0, idx_best)
    # Show high res version given a generator
    generator_cpu = copy.deepcopy(generator)
    generator_cpu = generator_cpu.to('cpu')
    generator_cpu.tensors_to('cpu')

    params = {'gamma': gamma}
    with torch.no_grad():
      img_high_res = generator_cpu.forward(params)
    img = img_high_res.detach().cpu().numpy()[0]

    img = np.clip(img, 0.0, 1.0)
    if save or show:
      # Swap Red with Blue
      img = img[...,[2, 1, 0]]
      img = np.clip(img, 0.0, 1.0) * 255
    if save:
      image_filename = f"{self._output_dir}/{self._file_basename}.png"
      cv2.imwrite(image_filename, img)
    if show:
      cv2_imshow(img)
      cv2.waitKey()
    return img

  def finish(self):
    """Finish video writing and save all other data."""
    if self._losses_history:
      losses_filename = f"{self._output_dir}/{self._file_basename}_losses"
      plot_and_save_losses(self._losses_history,
                           title=f"{self._file_basename} Losses",
                           filename=losses_filename)
    if self._video_steps:
      self._video_writer.close()
    if self._population_video:
      self._population_video_writer.close()
    metadata_filename = f"{self._output_dir}/{self._file_basename}_metadata.py"
    export_metadata(metadata_filename)

  def _add_video_frames(self, img_batch, losses):
    """Add images from numpy image batch to video writers.

    Args:
      img_batch: numpy array, batch of images (S,H,W,C)
      losses: numpy array, losses for each generator (S,N)
    """
    if self._video_steps and self._step % self._video_steps == 0:
      # Write image to video.
      best_img = img_batch[np.argmin(losses)]
      self._video_writer.add(cv2.resize(
          best_img, (best_img.shape[1] * 3, best_img.shape[0] * 3)))
      if self._population_video:
        laid_out = layout_img_batch(img_batch)
        self._population_video_writer.add(cv2.resize(
            laid_out, (laid_out.shape[1] * 2, laid_out.shape[0] * 2)))

#@title CollageTiler class

class CollageTiler():
  def __init__(self,
               wide, high,
               prompts,
               segmented_data,
               segmented_data_high_res,
               fixed_background_image,
               background_use,
               compositional,
               high_res_multiplier,
               output_dir,
               file_basename,
               video_steps,
               use_normalised_clip,
               use_image_augmentations,
               use_evolution,
               evolution_frequency,
               population_size,
               initial_search_size,
               optimisation_steps
               ):
    """Creates a large collage by producing multiple interlaced collages.

    Args:
      width: number of tiles wide
      height: number of tiles high
      prompts: list of prompts for the collage maker
      segmented_data: patch data for collage maker to use during opmtimisation
      segmented_data_high_res: high res patch data for final renders
      fixed_background_image: highest res background image
      background_use: how to use the background, e.g. per tile or whole image
      compositional: bool, use compositional for multi-CLIP collage tiles
      output_dir: directory for generated files
      video_steps: How often to capture frames for videos. Zero=never
      use_normalised_clip: bool, colour-correct images for CLIP evaluation
      use_image_augmentations: bool, produce image augmentations for evaluation
      use_evolution: bool, turn evolution on/off
      evolution_frequency: bool, how many steps between evolution evaluations
      population_size: int, size of population being evolvec
      initial_search_size: int, samples evaluated for each intial network
      optimisation_steps: int, training steps for the collage
    """
    self._tiles_wide = wide
    self._tiles_high = high
    self._prompts = prompts
    self._segmented_data = segmented_data
    self._segmented_data_high_res = segmented_data_high_res
    self._fixed_background_image = fixed_background_image
    self._background_use = background_use
    self._compositional_image = compositional
    self._high_res_multiplier = high_res_multiplier
    self._output_dir = output_dir
    self._video_steps = video_steps

    self._use_normalised_clip = use_normalised_clip
    self._use_image_augmentations = use_image_augmentations
    self._use_evolution = use_evolution
    self._evolution_frequency = evolution_frequency
    self._population_size = population_size
    self._initial_search_size = initial_search_size
    self._optimisation_steps = optimisation_steps

    pathlib.Path(self._output_dir).mkdir(parents=True, exist_ok=True)
    self._tile_basename = "tile_y{}_x{}{}"
    self._tile_width = 448 if self._compositional_image else 224
    self._tile_height = 448 if self._compositional_image else 224
    self._overlap = 1. / 3.

    # Size of bigger image
    self._width = self._tile_width * self._tiles_wide
    self._height = self._tile_height * self._tiles_high

    self._high_res_tile_width = self._tile_width * self._high_res_multiplier
    self._high_res_tile_height = self._tile_height * self._high_res_multiplier
    self._high_res_width = self._high_res_tile_width * self._tiles_wide
    self._high_res_height = self._high_res_tile_height * self._tiles_high

    self._print_info()
    self._x = 0
    self._y = 0
    self._collage_maker = None
    self._fixed_background = self._scale_fixed_background(high_res=True)

  def _print_info(self):
    print(f"Tiling {self._tiles_wide}x{self._tiles_high} collages")
    print("Optimisation:")
    print(f"Tile size: {self._tile_width}x{self._tile_height}")
    print(f"Global size: {self._width}x{self._height} (WxH)")
    print("High res:")
    print(
        f"Tile size: {self._high_res_tile_width}x{self._high_res_tile_height}")
    print(f"Global size: {self._high_res_width}x{self._high_res_height} (WxH)")
    for i, tile_prompts in enumerate(self._prompts):
      print(f"Tile {i} prompts: {tile_prompts}")

  def loop(self):
    while self._y < self._tiles_high:
      while self._x < self._tiles_wide:
        if not self._collage_maker:
          # Create new collage maker with its unique background.
          print(f"New collage creator for y{self._y}, x{self._x} with bg:")
          tile_bg, self._tile_high_res_bg = self._get_tile_background()
          show_and_save(tile_bg, img_format="SCHW", stitch=False, show=GUI)
          prompts = self._prompts[self._y * self._tiles_wide + self._x]
          self._collage_maker = CollageMaker(
              prompts=prompts,
              segmented_data=self._segmented_data,
              background_image=tile_bg,
              compositional_image=self._compositional_image,
              output_dir=self._output_dir,
              file_basename=self._tile_basename.format(self._y, self._x, ""),
              video_steps=self._video_steps,
              population_video=False,
              use_normalised_clip=self._use_normalised_clip,
              use_image_augmentations=self._use_image_augmentations,
              use_evolution=self._use_evolution,
              evolution_frequency=self._evolution_frequency,
              population_size=self._population_size,
              initial_search_size=self._initial_search_size,
              optimisation_steps=self._optimisation_steps
              )
        self._collage_maker.loop()
        collage_img = self._collage_maker.high_res_render(
            self._segmented_data_high_res,
            self._tile_high_res_bg,
            gamma=1.0,
            show=GUI,
            save=True)
        self._save_tile(collage_img / 255)
        # TODO: Currently calling finish will save video and download zip which is not needed.
        # self._collage_maker.finish()
        del self._collage_maker
        self._collage_maker = None
        self._x += 1
      self._y += 1
      self._x = 0
    return collage_img  # SHWC

  def _save_tile(self, img):
    print(type(img))
    background_image_np = np.asarray(img)
    background_image_np = background_image_np[..., ::-1].copy()
    filename = self._tile_basename.format(self._y, self._x, ".npy")
    np.save(f"{self._output_dir}/{filename}", background_image_np)

  def _scale_fixed_background(self, high_res=True):
    if self._fixed_background_image is None:
      return None
    multiplier = self._high_res_multiplier if high_res else 1
    if self._background_use == "Local":
      height = self._tile_height * multiplier
      width = self._tile_width * multiplier
    elif self._background_use == "Global":
      height = self._height * multiplier
      width = self._width * multiplier
    return resize(self._fixed_background_image.astype(float), (height, width))

  def _get_tile_background(self):
    """Get the background for a particular tile.

    This involves getting bordering imagery from left, top left, above and top
    right, where appropriate.
    i.e. tile (1,1) shares overlap with (0,1), (0,2) and (1,0)
    (0,0), (0,1), (0,2), (0,3)
    (1,0), (1,1), (1,2), (1,3)
    (2,0), (2,1), (2,2), (2,3)

    Note that (0,0) is not needed as its contribution is already in (0,1)
    """
    if self._fixed_background is None:
      tile_border_bg = np.zeros((self._high_res_tile_height,
                                self._high_res_tile_width, 3))
    else:
      if self._background_use == "Local":
        tile_border_bg = self._fixed_background.copy()
      else:  # Crop out section for this tile.
        #orgin_y = self._y * self._high_res_tile_height - int(
        #    self._high_res_tile_height * 2 * self._overlap)
        orgin_y = self._y * (self._high_res_tile_height
                             - int(self._high_res_tile_height * self._overlap))
        orgin_x = self._x * (self._high_res_tile_width
                             - int(self._high_res_tile_width * self._overlap))
        #orgin_x = self._x * self._high_res_tile_width - int(
        #    self._high_res_tile_width * 2 * self._overlap)
        tile_border_bg = self._fixed_background[
            orgin_y : orgin_y + self._high_res_tile_height,
            orgin_x : orgin_x + self._high_res_tile_width, :]
    tile_idx = dict()
    if self._x > 0:
      tile_idx["left"] = (self._y, self._x - 1)
    if self._y > 0:
      tile_idx["above"] = (self._y - 1, self._x)
      if self._x < self._tiles_wide - 1:  # Penultimate on the row
        tile_idx["above_right"] = (self._y - 1, self._x + 1)

    # Get and insert bodering tile content in this order.
    if "above" in tile_idx:
      self._copy_overlap(tile_border_bg, "above", tile_idx["above"])
    if "above_right" in tile_idx:
      self._copy_overlap(tile_border_bg, "above_right", tile_idx["above_right"])
    if "left" in tile_idx:
      self._copy_overlap(tile_border_bg, "left", tile_idx["left"])

    background_image = self._resize_image_for_torch(
        tile_border_bg, self._tile_height, self._tile_width)
    background_image_high_res = self._resize_image_for_torch(
        tile_border_bg,
        self._high_res_tile_height,
        self._high_res_tile_width).to('cpu')

    return background_image, background_image_high_res

  def _resize_image_for_torch(self, img, height, width):
    # Resize and permute to format used by Collage class (SCHW).
    if CUDA:
      img = torch.tensor(resize(img.astype(float), (height, width))).cuda()
    else:
      img = torch.tensor(resize(img.astype(float), (height, width)))
    return img.permute(2, 0, 1).to(torch.float32)

  def _copy_overlap(self, target, location, tile_idx):
    # print(
    #     f"Copying overlap from {location} ({tile_idx}) for {self._y},{self._x}")
    big_height = self._high_res_tile_height
    big_width = self._high_res_tile_width
    pixel_overlap = int(big_width * self._overlap)

    filename = self._tile_basename.format(tile_idx[0], tile_idx[1], ".npy")
    # print(f"Loading tile {filename})
    source = np.load(f"{self._output_dir}/{filename}")
    if location == "above":
      target[0 : pixel_overlap, 0 : big_width, :] = source[
          big_height - pixel_overlap : big_height, 0 : big_width, :]
    if location == "left":
      target[:, 0 : pixel_overlap, :] = source[
          :, big_width - pixel_overlap : big_width, :]
    elif location == "above_right":
      target[0 : pixel_overlap, big_width - pixel_overlap : big_width, :] = source[
          big_height - pixel_overlap : big_height, 0 : pixel_overlap, :]

  def assemble_tiles(self):
    # Stitch together the whole image.
    big_height = self._high_res_tile_height
    big_width = self._high_res_tile_width
    full_height = int((big_height + 2 * big_height * self._tiles_high) / 3)
    full_width = int((big_width + 2 * big_width * self._tiles_wide) / 3)
    full_image = np.zeros((full_height, full_width, 3)).astype('float32')

    for y in range(self._tiles_high):
      for x in range(self._tiles_wide):
        filename = self._tile_basename.format(y, x, ".npy")
        tile = np.load(f"{self._output_dir}/{filename}")
        y_offset = int(big_height * y * 2 / 3)
        x_offset = int(big_width * x * 2 / 3)
        full_image[y_offset : y_offset + big_height,
                   x_offset : x_offset + big_width, :] = tile[:, :, :]
    filename = f"final_tiled_image"
    print(f"Saving assembled tiles to {filename}")
    show_and_save(full_image, img_format="SHWC", stitch=False,
        filename=filename, show=GUI)

"""# Make Collages"""

VIDEO_STEPS =   500#@param {type:"integer"}
TRACE_EVERY =   500#@param {type:"integer"}

PROMPTS = [GLOBAL_PROMPT]
USE_SOLID_COLOUR_BACKGROUND = BACKGROUND == "Solid colour below"
if BACKGROUND == "None (black)":
  # 'No background' is actually a black background.
  USE_SOLID_COLOUR_BACKGROUND = True
  BACKGROUND_RED = 0
  BACKGROUND_GREEN = 0
  BACKGROUND_BLUE = 0


# Turn off tiling if either boolean is set or width/height set to 1.
if not TILE_IMAGES or (TILES_WIDE == 1 and TILES_HIGH == 1):
  TILES_WIDE = 1
  TILES_HIGH = 1
  TILE_IMAGES = False


if not TILE_IMAGES or GLOBAL_TILE_PROMPT:
  TILE_PROMPTS = [GLOBAL_PROMPT] * TILES_HIGH * TILES_WIDE
else:
  TILE_PROMPTS = []
  count_y = 0
  count_x = 0
  for row in TILE_PROMPT_STRING.split("/"):
    for prompt in row.split("|"):
      prompt = prompt.strip()
      TILE_PROMPTS.append(prompt)
      count_x += 1
    if count_x != TILES_WIDE:
      raise ValueError(f"Insufficient prompts for row {count_y}; expected {TILES_WIDE} but got {count_x}")
    count_x = 0
    count_y += 1
  if count_y != TILES_HIGH:
    raise ValueError(f"Insufficient prompt rows; expected {TILES_HIGH} but got {count_y}")

print("Tile prompts: ", TILE_PROMPTS)

tile_count = 0
all_prompts = []
for y in range(TILES_HIGH):
  for x in range(TILES_WIDE):
    tile_prompts = []
    if COMPOSITIONAL_IMAGE:
      if TILE_IMAGES:
        tile_prompts = [
            TILE_PROMPT_FORMATING.format(TILE_PROMPTS[tile_count])] * 9
      else:
        tile_prompts = [PROMPT_x0_y0, PROMPT_x1_y0, PROMPT_x2_y0,
                        PROMPT_x0_y1, PROMPT_x1_y1, PROMPT_x2_y1,
                        PROMPT_x0_y2, PROMPT_x1_y2, PROMPT_x2_y2]
    tile_prompts.append(TILE_PROMPTS[tile_count])
    tile_count += 1
    all_prompts.append(tile_prompts)
print(f"All prompts: {all_prompts}")

#@title Get background image (if using one)

def upload_files():
  # Upload and save to Colab's disk.
  uploaded = files.upload()
  # Save to disk
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())

def load_image(filename, as_cv2_image=False, show=False):
  # Load an image as [0,1] RGB numpy array or cv2 image format.
  img = cv2.imread(filename)
  if show:
    cv2_imshow(img)
  if as_cv2_image:
    return img  # With colour format BGR
  img = np.asarray(img)
  return img[..., ::-1] / 255.  # Reverse colour dim to convert BGR to RGB

background_image = None
if USE_SOLID_COLOUR_BACKGROUND:
  background_image = np.ones((10, 10, 3), dtype=np.float32)
  background_image[:, :, 0] = BACKGROUND_RED
  background_image[:, :, 1] = BACKGROUND_GREEN
  background_image[:, :, 2] = BACKGROUND_BLUE
  # background_image[:, :, 3] = 255.
  print('Defined background colour ({}, {}, {})'.format(
      BACKGROUND_RED, BACKGROUND_GREEN, BACKGROUND_BLUE))
else:
  backgrounds = upload_files()
  background_image = load_image(backgrounds[0], show=GUI)

#@title Create collage! (Initialisation)
ct = CollageTiler(wide=TILES_WIDE,
                  high=TILES_HIGH,
                  prompts=all_prompts,
                  segmented_data=segmented_data,
                  segmented_data_high_res=segmented_data_high_res,
                  fixed_background_image=background_image,
                  background_use=BACKGROUND_USE,
                  compositional=COMPOSITIONAL_IMAGE,
                  high_res_multiplier=MULTIPLIER_BIG_IMAGE,
                  output_dir=OUTPUT_DIR,
                  file_basename='collage_tiler',
                  video_steps=0,
                  use_normalised_clip=USE_NORMALIZED_CLIP,
                  use_image_augmentations=USE_IMAGE_AUGMENTATIONS,
                  use_evolution=USE_EVOLUTION,
                  evolution_frequency=EVOLUTION_FREQUENCY,
                  population_size=POP_SIZE,
                  initial_search_size=INITIAL_SEARCH_SIZE,
                  optimisation_steps=OPTIM_STEPS
                  )

#@markdown To edit patches interrupt this cell and run the one below this. Re-run this cell afterwards to continue generating the image.
output = ct.loop()

#@title Render high res image and finish up.
ct.assemble_tiles()

if CLEAN_UP:
  for file_match in ["*.npy", "tile_*.png"]:
    files = glob.glob(f"{OUTPUT_DIR}/{file_match}")
    for f in files:
        os.remove(f)

