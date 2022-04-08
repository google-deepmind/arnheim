"""Collage network definition.

Arnheim 3 - Collage
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

import copy
from . import rendering
from . import transformations
import numpy as np
import torch


class PopulationCollage(torch.nn.Module):
  """Population-based segmentation collage network.

  Image structure in this class is SCHW.
  """

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
    self._canvas_height = config["canvas_height"]
    self._canvas_width = config["canvas_width"]
    self._high_res_multiplier = config["high_res_multiplier"]
    self._num_patches = self.config["num_patches"]
    self._pop_size = pop_size
    requires_grad = not is_high_res

    # Create the spatial transformer and colour transformer for patches.
    self.spatial_transformer = transformations.PopulationAffineTransforms(
        config, device, num_patches=self._num_patches, pop_size=pop_size,
        requires_grad=requires_grad, is_high_res=is_high_res)
    if self.config["colour_transformations"] == "HSV space":
      self.colour_transformer = transformations.PopulationColourHSVTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size,
          requires_grad=requires_grad)
    elif self.config["colour_transformations"] == "RGB space":
      self.colour_transformer = transformations.PopulationColourRGBTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size,
          requires_grad=requires_grad)
    else:
      self.colour_transformer = transformations.PopulationOrderOnlyTransforms(
          config, device, num_patches=self._num_patches, pop_size=pop_size,
          requires_grad=requires_grad)
    if config["torch_device"] == "cuda":
      self.spatial_transformer = self.spatial_transformer.cuda()
      self.colour_transformer = self.colour_transformer.cuda()
    self.coloured_patches = None

    # Optimisation is run in low-res, final rendering is in high-res.
    self._high_res = is_high_res

    # Store the background image (low- and high-res).
    self.background_image = background_image
    if self.background_image is not None:
      print(f"Background image of size {self.background_image.shape}")

    # Store the dataset (low- and high-res).
    self._dataset = segmented_data
    # print(f"There are {len(self._dataset)} image patches in the dataset")

    # Initial set of indices pointing to self._num_patches first dataset images.
    self.patch_indices = [np.arange(self._num_patches) % len(self._dataset)
                          for _ in range(pop_size)]

    # Patches in low and high-res, will be initialised on demand.
    self.patches = None

  def store_patches(self, population_idx=None):
    """Store the image patches for each population element."""
    if self._high_res:
      for _ in range(20):
        print("NOT STORING HIGH-RES PATCHES")
      return

    if population_idx is not None and self.patches is not None:
      list_indices_population = [population_idx]
      self.patches[population_idx, :, :4, :, :] = 0
    else:
      list_indices_population = np.arange(self._pop_size)
      self.patches = torch.zeros(
          self._pop_size, self._num_patches, 5, self._canvas_height,
          self._canvas_width).to(self.device)

    # Put the segmented data into the patches.
    for i in list_indices_population:
      for j in range(self._num_patches):
        patch_i_j = self._fetch_patch(i, j, self._high_res)
        self.patches[i, j, ...] = patch_i_j

  def _fetch_patch(self, idx_population, idx_patch, is_high_res):
    """Helper function to fetch a patch and store on the whole canvas."""
    k = self.patch_indices[idx_population][idx_patch]
    patch_j = torch.tensor(
        self._dataset[k].swapaxes(0, 2) / 255.0).to(self.device)
    width_j = patch_j.shape[1]
    height_j = patch_j.shape[2]
    if is_high_res:
      w0 = int((self._canvas_width * self._high_res_multiplier - width_j)
               / 2.0)
      h0 = int((self._canvas_height * self._high_res_multiplier - height_j)
               / 2.0)
      mapped_patch = torch.zeros(
          5,
          self._canvas_height * self._high_res_multiplier,
          self._canvas_width * self._high_res_multiplier
          ).to("cpu")
    else:
      w0 = int((self._canvas_width - width_j) / 2.0)
      h0 = int((self._canvas_height - height_j) / 2.0)
      mapped_patch = torch.zeros(
          5, self._canvas_height, self._canvas_width).to(self.device)
    mapped_patch[4, :, :] = 1.0
    mapped_patch[:4, w0:(w0 + width_j), h0:(h0 + height_j)] = patch_j
    return mapped_patch

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      # Copy the patches indices from the parent to the child.
      self.patch_indices[child] = copy.deepcopy(self.patch_indices[parent])

      # Mutate the child patches with a single swap from the original dataset.
      if self.config["patch_mutation_probability"] > np.random.uniform():
        idx_dataset = np.random.randint(len(self._dataset))
        idx_patch = np.random.randint(self._num_patches)
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
      self.spatial_transformer.copy_from(
          other.spatial_transformer, idx_to, idx_from)
      self.colour_transformer.copy_from(
          other.colour_transformer, idx_to, idx_from)
      if not self._high_res:
        self.store_patches(idx_to)

  def forward(self, params=None):
    """Input-less forward function."""

    assert not self._high_res
    if self.patches is None:
      self.store_patches()
    shifted_patches = self.spatial_transformer(self.patches)
    background_image = self.background_image
    if params is not None and "no_background" in params:
      print("Not using background_image")
      background_image = None

    self.coloured_patches = self.colour_transformer(shifted_patches)
    if self.config["render_method"] == "transparency":
      img = rendering.population_render_transparency(
          self.coloured_patches,
          invert_colours=self.config["invert_colours"], b=background_image)
    elif self.config["render_method"] == "masked_transparency_clipped":
      img = rendering.population_render_masked_transparency(
          self.coloured_patches, mode="clipped",
          invert_colours=self.config["invert_colours"], b=background_image)
    elif self.config["render_method"] == "masked_transparency_normed":
      img = rendering.population_render_masked_transparency(
          self.coloured_patches, mode="normed",
          invert_colours=self.config["invert_colours"], b=background_image)
    elif self.config["render_method"] == "opacity":
      img = rendering.population_render_overlap(
          self.coloured_patches,
          invert_colours=self.config["invert_colours"], b=background_image)
    else:
      print("Unhandled render method")
    if params is not None and "no_background" in params:
      print("Setting alpha to zero outside of patches")
      mask = self.coloured_patches[:, :, 3:4, :, :].sum(1) > 0
      mask = mask.permute(0, 2, 3, 1)
      img = torch.concat([img, mask], axis=-1)
    return img

  def forward_high_res(self, params=None):
    """Input-less forward function."""

    assert self._high_res

    max_render_size = params.get("max_block_size_high_res", 1000)
    w = self._canvas_width * self._high_res_multiplier
    h = self._canvas_height * self._high_res_multiplier
    if (self._high_res_multiplier % 8 == 0 and
        self._canvas_width * 8 < max_render_size and
        self._canvas_height * 8 < max_render_size):
      num_w = int(self._high_res_multiplier / 8)
      num_h = int(self._high_res_multiplier / 8)
      delta_w = self._canvas_width * 8
      delta_h = self._canvas_height * 8
    elif (self._high_res_multiplier % 4 == 0 and
          self._canvas_width * 4 < max_render_size and
          self._canvas_height * 4 < max_render_size):
      num_w = int(self._high_res_multiplier / 4)
      num_h = int(self._high_res_multiplier / 4)
      delta_w = self._canvas_width * 4
      delta_h = self._canvas_height * 4
    elif (self._high_res_multiplier % 2 == 0 and
          self._canvas_width * 2 < max_render_size and
          self._canvas_height * 2 < max_render_size):
      num_w = int(self._high_res_multiplier / 2)
      num_h = int(self._high_res_multiplier / 2)
      delta_w = self._canvas_width * 2
      delta_h = self._canvas_height * 2
    else:
      num_w = self._high_res_multiplier
      num_h = self._high_res_multiplier
      delta_w = self._canvas_width
      delta_h = self._canvas_height

    img = torch.zeros((1, h, w, 4))
    img[..., 3] = 1.0

    background_image = self.background_image
    if params is not None and "no_background" in params:
      print("Not using background_image")
      background_image = None

    for u in range(num_w):
      for v in range(num_h):
        x0 = u * delta_w
        x1 = (u + 1) * delta_w
        y0 = v * delta_h
        y1 = (v + 1) * delta_h
        print(f"[{u}, {v}] idx [{x0}:{x1}], [{y0}:{y1}]")

        # Extract full patches, apply spatial transform individually and crop.
        shifted_patches_uv = []
        for idx_patch in range(self._num_patches):
          patch = self._fetch_patch(0, idx_patch, True).unsqueeze(0)
          patch_uv = self.spatial_transformer(patch, idx_patch)
          patch_uv = patch_uv[:, :, :, y0:y1, x0:x1]
          shifted_patches_uv.append(patch_uv)
        shifted_patches_uv = torch.cat(shifted_patches_uv, 1)

        # Crop background?
        if background_image is not None:
          background_image_uv = background_image[:, y0:y1, x0:x1]
        else:
          background_image_uv = None

        # Appy colour transform and render.
        coloured_patches_uv = self.colour_transformer(shifted_patches_uv)
        if self.config["render_method"] == "transparency":
          img_uv = rendering.population_render_transparency(
              coloured_patches_uv,
              invert_colours=self.config["invert_colours"],
              b=background_image_uv)
        elif self.config["render_method"] == "masked_transparency_clipped":
          img_uv = rendering.population_render_masked_transparency(
              coloured_patches_uv, mode="clipped",
              invert_colours=self.config["invert_colours"],
              b=background_image_uv)
        elif self.config["render_method"] == "masked_transparency_normed":
          img_uv = rendering.population_render_masked_transparency(
              coloured_patches_uv, mode="normed",
              invert_colours=self.config["invert_colours"],
              b=background_image_uv)
        elif self.config["render_method"] == "opacity":
          img_uv = rendering.population_render_overlap(
              coloured_patches_uv,
              invert_colours=self.config["invert_colours"],
              b=background_image_uv)
        else:
          print("Unhandled render method")

        if params is not None and "no_background" in params:
          print("Setting alpha to zero outside of patches")
          mask_uv = coloured_patches_uv[:, :, 3:4, :, :].sum(1) > 0
          mask_uv = mask_uv.permute(0, 2, 3, 1)
          img_uv = torch.concat([img_uv, mask_uv], axis=-1)
          img[0, y0:y1, x0:x1, :4] = img_uv
        else:
          img[0, y0:y1, x0:x1, :3] = img_uv
        print(f"Finished [{u}, {v}] idx [{x0}:{x1}], [{y0}:{y1}]")

    print(img.size())
    return img

  def tensors_to(self, device):
    self.spatial_transformer.tensor_to(device)
    self.colour_transformer.tensor_to(device)
    if self.patches is not None:
      self.patches = self.patches.to(device)
