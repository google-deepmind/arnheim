# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Arnheim 3 - Collage
# Piotr Mirowski, Dylan Banarse, Mateusz Malinowski, Yotam Doron, Oriol Vinyals,
# Simon Osindero, Chrisantha Fernando
# DeepMind, 2021-2022

# Command-line version of the Google Colab code available at:
# https://github.com/deepmind/arnheim/blob/main/arnheim_3.ipynb

# Loading and processing collage patches.

import io
import os
import pathlib
import requests

import cv2
import numpy as np


SHOW_PATCHES = True


def add_binary_alpha_mask(patch):
  """Black pixels treated as having alpha=0, all other pixels have alpha=255"""
  shape = patch.shape
  mask = ((patch.sum(2) > 0) * 255).astype(np.uint8)
  return np.concatenate([patch, np.expand_dims(mask, -1)], axis=-1)


def resize_patch(patch, coeff):
  return cv2.resize(patch.astype(float),
                    (int(np.round(patch.shape[1] * coeff)),
                     int(np.round(patch.shape[0] * coeff))))


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
  print(f"{len(segmented_data)} patches, max {shape_max}, min {shape_min}\n")


def cached_url_download(url):
  cache_filename = os.path.basename(url)
  cache = pathlib.Path(cache_filename)
  if not cache.is_file():
    print(f"Downloading {cache_filename} from {url}")
    r = requests.get(url)
    bytesio_object = io.BytesIO(r.content)
    with open(cache_filename, "wb") as f:
      f.write(bytesio_object.getbuffer())
  else:
    print("Using cached version of " + cache_filename)
  return np.load(cache, allow_pickle=True)


def get_segmented_data_initial(config):

  if len(config["url_to_patch_file"]) > 0:
    segmented_data_initial = cached_url_download(config["url_to_patch_file"])
  else:
    repo_file = config["patch_set"]
    repo_root = config["patch_repo_root"]
    segmented_data_initial = cached_url_download(
        f"{repo_root}/collage_patches/{repo_file}")

  segmented_data_initial_tmp = []
  for i in range(len(segmented_data_initial)):
    if segmented_data_initial[i].shape[2] == 3:
      segmented_data_initial_tmp.append(add_binary_alpha_mask(
          segmented_data_initial[i]))
    else:
      segmented_data_initial_tmp.append(
          segmented_data_initial[i])

  segmented_data_initial = segmented_data_initial_tmp
  return segmented_data_initial


def normalise_patch_brightness(patch):
  max_intensity = max(patch.max(), 1.0)
  return ((patch / max_intensity) * 255).astype(np.uint8)


def get_segmented_data(config):

  segmented_data_initial = get_segmented_data_initial(config)

  # Permute the order of the segmented images.
  num_patches = len(segmented_data_initial)
  order = np.random.permutation(num_patches)

  # Compress all images until they are at most 1/PATCH_MAX_PROPORTION of the
  # large canvas size.
  canvas_height = config['canvas_height']
  canvas_width = config['canvas_width']
  hires_height = canvas_height * config['high_res_multiplier']
  hires_width = canvas_width * config['high_res_multiplier']
  height_large_max = hires_height / config["patch_max_proportion"]
  width_large_max = hires_width / config["patch_max_proportion"]
  if config["fixed_scale_patches"]:
    print(f"Max size for fixed scale patches: ({hires_height},{hires_width})")
  else:
    print(
        f"Max patch size on large img: ({height_large_max}, {width_large_max})")
  segmented_data = []
  segmented_data_high_res = []
  for patch_i in range(num_patches):
    segmented_data_initial_i = segmented_data_initial[
        order[patch_i]].astype(np.float32).swapaxes(0, 1)
    shape_i = segmented_data_initial_i.shape
    h_i = shape_i[0]
    w_i = shape_i[1]
    if h_i >= config["patch_height_min"] and w_i >= config["patch_width_min"]:
      # Coefficient for resizing the patch.
      if config["fixed_scale_patches"]:
        coeff_i_large = config["fixed_scale_coeff"]
        if h_i * coeff_i_large > hires_height:
          coeff_i_large = hires_height / h_i
        if w_i * coeff_i_large > width_large_max:
          coeff_i_large = min(coeff_i_large, hires_width / w_i)
        if coeff_i_large != config["fixed_scale_coeff"]:
          print(
              f"Patch {patch_i} too large; scaled to {coeff_i_large:.2f}")
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
      coeff_i = coeff_i_large / config['high_res_multiplier']
      segmented_data_i = resize_patch(segmented_data_initial_i, coeff_i)
      shape_i = segmented_data_i.shape
      if (shape_i[0] > canvas_height
          or shape_i[1] > config['canvas_width']):

        print(f"{shape_i} exceeds canvas ({canvas_height},{canvas_width})")
        import pdb; pdb.set_trace()
      if config["normalize_patch_brightness"]:
        segmented_data_i[...,:3] = normalise_patch_brightness(
            segmented_data_i[...,:3])
        segmented_data_high_res_i[...,:3] = normalise_patch_brightness(
            segmented_data_high_res_i[...,:3])
      segmented_data_high_res_i = segmented_data_high_res_i.astype(np.uint8)
      segmented_data_high_res.append(segmented_data_high_res_i)
      segmented_data_i = segmented_data_i.astype(np.uint8)
      segmented_data.append(segmented_data_i)
    else:
      print(f"Discard patch of size {h_i}x{w_i}")

  if SHOW_PATCHES:
    print("Patch sizes during optimisation:")
    print_size_segmented_data(segmented_data, show=config["gui"])
    print("Patch sizes for high-resolution final image:")
    print_size_segmented_data(segmented_data_high_res, show=config["gui"])

  return segmented_data, segmented_data_high_res
