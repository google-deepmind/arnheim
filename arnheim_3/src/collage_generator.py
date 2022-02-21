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

# Collage network definition.

import copy
import time

import numpy as np
import torch

from . import rendering
from . import transformations


class PopulationCollage(torch.nn.Module):
  """Population-based segmentation collage network.
  Image structure in this class is SCHW."""
  def __init__(self,
               config,
               device,
               pop_size=1,
               is_high_res=False,
               segmented_data=None,
               background_image=None):
    """Constructor, relying on global parameters."""
    super(PopulationCollage, self).__init__()

    # Config, device, number of patches and population size.
    self.config = config
    self.device = device
    self._canvas_height = config['canvas_height']
    self._canvas_width = config['canvas_width']
    self._high_res_multiplier = config['high_res_multiplier']
    self._num_patches = self.config['num_patches']
    self._pop_size = pop_size

    # Create the spatial transformer and colour transformer for patches.
    self.spatial_transformer = transformations.PopulationAffineTransforms(
        config, device, num_patches=self._num_patches, pop_size=pop_size)
    if self.config['colour_transformations'] == "HSV space":
      self.colour_transformer = transformations.PopulationColourHSVTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size)
    elif self.config['colour_transformations'] == "RGB space":
      self.colour_transformer = transformations.PopulationColourRGBTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size)
    else:
      self.colour_transformer = transformations.PopulationOrderOnlyTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size)
    if config["torch_device"] == "cuda":
      self.spatial_transformer = self.spatial_transformer.cuda()
      self.colour_transformer = self.colour_transformer.cuda()
    self.coloured_patches = None

    # Optimisation is run in low-res, final rendering is in high-res.
    self._high_res = is_high_res

    # Store the background image (low- and high-res).
    self.background_image = background_image
    if self.background_image is not None:
      print(f'Background image of size {self.background_image.shape}')

    # Store the dataset (low- and high-res).
    self._dataset = segmented_data
    #print(f'There are {len(self._dataset)} image patches in the dataset')

    # Initial set of indices, pointing to the NUM_PATCHES first dataset images.
    self.patch_indices = [np.arange(self._num_patches) % len(self._dataset)
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
      if self._high_res:
        self.patches = torch.zeros(
            1, self._num_patches, 5,
            self._canvas_height * self._high_res_multiplier,
            self._canvas_width * self._high_res_multiplier
            ).to('cpu')
      else:
        self.patches = torch.zeros(
            self._pop_size, self._num_patches, 5, self._canvas_height,
            self._canvas_width).to(self.device)
      self.patches[:, :, 4, :, :] = 1.0

    # Put the segmented data into the patches.
    for i in list_indices:
      for j in range(self._num_patches):
        k = self.patch_indices[i][j]
        patch_j = torch.tensor(
            self._dataset[k].swapaxes(0, 2) / 255.0).to(self.device)
        width_j = patch_j.shape[1]
        height_j = patch_j.shape[2]
        if self._high_res:
          w0 = int((self._canvas_width * self._high_res_multiplier - width_j)
                   / 2.0)
          h0 = int((self._canvas_height * self._high_res_multiplier - height_j)
                   / 2.0)
        else:
          w0 = int((self._canvas_width - width_j) / 2.0)
          h0 = int((self._canvas_height - height_j) / 2.0)
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
      if self.config['patch_mutation_probability'] > np.random.uniform():
        idx_dataset  = np.random.randint(len(self._dataset))
        idx_patch  = np.random.randint(self._num_patches)
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
    background_image = self.background_image
    self.coloured_patches = self.colour_transformer(shifted_patches)
    if self.config['render_method'] == "transparency":
      img = rendering.population_render_transparency(self.coloured_patches,
          invert_colours=self.config['invert_colours'], b=background_image)
    elif self.config['render_method'] == "masked_transparency_clipped":
      img = rendering.population_render_masked_transparency(
          self.coloured_patches, mode="clipped",
          invert_colours=self.config['invert_colours'], b=background_image)
    elif self.config['render_method'] == "masked_transparency_normed":
      img = rendering.population_render_masked_transparency(
          self.coloured_patches, mode="normed",
          invert_colours=self.config['invert_colours'], b=background_image)
    elif self.config['render_method'] == "opacity":
      if params is not None and 'gamma' in params:
        gamma = params['gamma']
      else:
        gamma = None
      img = rendering.population_render_overlap(self.coloured_patches,
          invert_colours=self.config['invert_colours'], b=background_image)
    else:
      print("Unhandled render method")
    return img

  def tensors_to(self, device):
    self.spatial_transformer.tensor_to(device)
    self.colour_transformer.tensor_to(device)
    self.patches = self.patches.to(device)
