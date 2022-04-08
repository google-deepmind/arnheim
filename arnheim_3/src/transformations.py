"""Colour and affine transform classes.

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

from kornia.color import hsv

import numpy as np
import torch
import torch.nn.functional as F


class PopulationAffineTransforms(torch.nn.Module):
  """Population-based Affine Transform network."""

  def __init__(self, config, device, num_patches=1, pop_size=1,
               requires_grad=True, is_high_res=False):
    super(PopulationAffineTransforms, self).__init__()

    self.config = config
    self.device = device
    self._pop_size = pop_size
    self._is_high_res = is_high_res
    print('PopulationAffineTransforms is_high_res={}, requires_grad={}'.format(
        self._is_high_res, requires_grad))

    self._min_rot = self.config['min_rot_deg'] * np.pi / 180.
    self._max_rot = self.config['max_rot_deg'] * np.pi / 180.
    matrices_translation = (
        (np.random.rand(pop_size, num_patches, 2, 1)
         * (self.config['max_trans_init'] - self.config['min_trans_init']))
        + self.config['min_trans_init'])
    matrices_rotation = (
        (np.random.rand(pop_size, num_patches, 1, 1)
         * (self._max_rot - self._min_rot)) + self._min_rot)
    matrices_scale = (
        (np.random.rand(pop_size, num_patches, 1, 1)
         * (self.config['max_scale'] - self.config['min_scale']))
        + self.config['min_scale'])
    matrices_squeeze = (
        (np.random.rand(pop_size, num_patches, 1, 1) * (
            (self.config['max_squeeze'] - self.config['min_squeeze'])
            + self.config['min_squeeze'])))
    matrices_shear = (
        (np.random.rand(pop_size, num_patches, 1, 1)
         * (self.config['max_shear'] - self.config['min_shear']))
        + self.config['min_shear'])
    self.translation = torch.nn.Parameter(
        torch.tensor(matrices_translation, dtype=torch.float),
        requires_grad=requires_grad)
    self.rotation = torch.nn.Parameter(
        torch.tensor(matrices_rotation, dtype=torch.float),
        requires_grad=requires_grad)
    self.scale = torch.nn.Parameter(
        torch.tensor(matrices_scale, dtype=torch.float),
        requires_grad=requires_grad)
    self.squeeze = torch.nn.Parameter(
        torch.tensor(matrices_squeeze, dtype=torch.float),
        requires_grad=requires_grad)
    self.shear = torch.nn.Parameter(
        torch.tensor(matrices_shear, dtype=torch.float),
        requires_grad=requires_grad)
    self._identity = (
        torch.ones((pop_size, num_patches, 1, 1)) * torch.eye(2).unsqueeze(0)
        ).to(self.device)
    self._zero_column = torch.zeros(
        (pop_size, num_patches, 2, 1)).to(self.device)
    self._unit_row = (
        torch.ones((pop_size, num_patches, 1, 1)) * torch.tensor([0., 0., 1.])
        ).to(self.device)
    self._zeros = torch.zeros((pop_size, num_patches, 1, 1)).to(self.device)

  def _clamp(self):
    self.translation.data = self.translation.data.clamp(
        min=self.config['min_trans'], max=self.config['max_trans'])
    self.rotation.data = self.rotation.data.clamp(
        min=self._min_rot, max=self._max_rot)
    self.scale.data = self.scale.data.clamp(
        min=self.config['min_scale'], max=self.config['max_scale'])
    self.squeeze.data = self.squeeze.data.clamp(
        min=self.config['min_squeeze'], max=self.config['max_squeeze'])
    self.shear.data = self.shear.data.clamp(
        min=self.config['min_shear'], max=self.config['max_shear'])

  def copy_and_mutate_s(self, parent, child):
    """Copy parameters to child, mutating transform parameters."""
    with torch.no_grad():
      self.translation[child, ...] = (
          self.translation[parent, ...]
          + self.config['pos_and_rot_mutation_scale'] * torch.randn(
              self.translation[child, ...].shape).to(self.device))
      self.rotation[child, ...] = (
          self.rotation[parent, ...]
          + self.config['pos_and_rot_mutation_scale'] * torch.randn(
              self.rotation[child, ...].shape).to(self.device))
      self.scale[child, ...] = (
          self.scale[parent, ...]
          + self.config['scale_mutation_scale'] * torch.randn(
              self.scale[child, ...].shape).to(self.device))
      self.squeeze[child, ...] = (
          self.squeeze[parent, ...]
          + self.config['distort_mutation_scale'] * torch.randn(
              self.squeeze[child, ...].shape).to(self.device))
      self.shear[child, ...] = (
          self.shear[parent, ...]
          + self.config['distort_mutation_scale'] * torch.randn(
              self.shear[child, ...].shape).to(self.device))

  def copy_from(self, other, idx_to, idx_from):
    """Copy parameters from other spatial transform, for selected indices."""
    assert idx_to < self._pop_size
    with torch.no_grad():
      self.translation[idx_to, ...] = other.translation[idx_from, ...]
      self.rotation[idx_to, ...] = other.rotation[idx_from, ...]
      self.scale[idx_to, ...] = other.scale[idx_from, ...]
      self.squeeze[idx_to, ...] = other.squeeze[idx_from, ...]
      self.shear[idx_to, ...] = other.shear[idx_from, ...]

  def forward(self, x, idx_patch=None):
    self._clamp()
    scale_affine_mat = torch.cat([
        torch.cat([self.scale, self.shear], 3),
        torch.cat([self._zeros, self.scale * self.squeeze], 3)], 2)
    scale_affine_mat = torch.cat([
        torch.cat([scale_affine_mat, self._zero_column], 3),
        self._unit_row], 2)
    rotation_affine_mat = torch.cat([
        torch.cat([torch.cos(self.rotation), -torch.sin(self.rotation)], 3),
        torch.cat([torch.sin(self.rotation), torch.cos(self.rotation)], 3)], 2)
    rotation_affine_mat = torch.cat([
        torch.cat([rotation_affine_mat, self._zero_column], 3),
        self._unit_row], 2)

    scale_rotation_mat = torch.matmul(scale_affine_mat,
                                      rotation_affine_mat)[:, :, :2, :]
    # Population and patch dimensions (0 and 1) need to be merged.
    # E.g. from (POP_SIZE, NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    # to (POP_SIZE * NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    if idx_patch is not None and self._is_high_res:
      scale_rotation_mat = scale_rotation_mat[:, idx_patch, :, :]
      num_patches = 1
    else:
      scale_rotation_mat = scale_rotation_mat[:, :, :2, :].view(
          1, -1, *(scale_rotation_mat[:, :, :2, :].size()[2:])).squeeze()
      num_patches = x.size()[1]
      x = x.view(1, -1, *(x.size()[2:])).squeeze()
    # print('scale_rotation_mat', scale_rotation_mat.size())
    # print('x', x.size())
    scaled_rotated_grid = F.affine_grid(
        scale_rotation_mat, x.size(), align_corners=True)
    scaled_rotated_x = F.grid_sample(x, scaled_rotated_grid, align_corners=True)

    translation_affine_mat = torch.cat([self._identity, self.translation], 3)
    if idx_patch is not None and self._is_high_res:
      translation_affine_mat = translation_affine_mat[:, idx_patch, :, :]
    else:
      translation_affine_mat = translation_affine_mat.view(
          1, -1, *(translation_affine_mat.size()[2:])).squeeze()
    # print('translation_affine_mat', translation_affine_mat.size())
    # print('scaled_rotated_x', scaled_rotated_x.size())
    translated_grid = F.affine_grid(
        translation_affine_mat, scaled_rotated_x.size(), align_corners=True)
    y = F.grid_sample(scaled_rotated_x, translated_grid, align_corners=True)
    # print('y', y.size())
    # print('num_patches', num_patches)
    return y.view(self._pop_size, num_patches, *(y.size()[1:]))

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


class PopulationOrderOnlyTransforms(torch.nn.Module):
  """No color transforms, just ordering of patches."""

  def __init__(self, config, device, num_patches=1, pop_size=1,
               requires_grad=True):
    super(PopulationOrderOnlyTransforms, self).__init__()

    self.config = config
    self.device = device
    self._pop_size = pop_size
    print(f'PopulationOrderOnlyTransforms requires_grad={requires_grad}')

    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=requires_grad)
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
  """HSV color transforms and ordering of patches."""

  def __init__(self, config, device, num_patches=1, pop_size=1,
               requires_grad=True):
    super(PopulationColourHSVTransforms, self).__init__()

    self.config = config
    self.device = device
    print('PopulationColourHSVTransforms for {} patches, {} individuals'.format(
        num_patches, pop_size))
    self._pop_size = pop_size
    self._min_hue = self.config['min_hue_deg'] * np.pi / 180.
    self._max_hue = self.config['max_hue_deg'] * np.pi / 180.
    print(f'PopulationColourHSVTransforms requires_grad={requires_grad}')

    coeff_hue = (0.5 * (self._max_hue - self._min_hue) + self._min_hue)
    coeff_sat = (0.5 * (self.config['max_sat'] - self.config['min_sat'])
                 + self.config['min_sat'])
    coeff_val = (0.5 * (self.config['max_val'] - self.config['min_val'])
                 + self.config['min_val'])
    population_hues = (np.random.rand(pop_size, num_patches, 1, 1, 1)
                       * coeff_hue)
    population_saturations = np.random.rand(
        pop_size, num_patches, 1, 1, 1) * coeff_sat
    population_values = np.random.rand(
        pop_size, num_patches, 1, 1, 1) * coeff_val
    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self.hues = torch.nn.Parameter(
        torch.tensor(population_hues, dtype=torch.float),
        requires_grad=requires_grad)
    self.saturations = torch.nn.Parameter(
        torch.tensor(population_saturations, dtype=torch.float),
        requires_grad=requires_grad)
    self.values = torch.nn.Parameter(
        torch.tensor(population_values, dtype=torch.float),
        requires_grad=requires_grad)
    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=requires_grad)
    self._hsv_to_rgb = hsv.HsvToRgb()

  def _clamp(self):
    self.hues.data = self.hues.data.clamp(
        min=self._min_hue, max=self._max_hue)
    self.saturations.data = self.saturations.data.clamp(
        min=self.config['min_sat'], max=self.config['max_sat'])
    self.values.data = self.values.data.clamp(
        min=self.config['min_val'], max=self.config['max_val'])
    self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      self.hues[child, ...] = (
          self.hues[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.hues[child, ...].shape).to(self.device))
      self.saturations[child, ...] = (
          self.saturations[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.saturations[child, ...].shape).to(self.device))
      self.values[child, ...] = (
          self.values[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.values[child, ...].shape).to(self.device))
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
  """RGB color transforms and ordering of patches."""

  def __init__(self, config, device, num_patches=1, pop_size=1,
               requires_grad=True):
    super(PopulationColourRGBTransforms, self).__init__()

    self.config = config
    self.device = device
    print('PopulationColourRGBTransforms for {} patches, {} individuals'.format(
        num_patches, pop_size))
    self._pop_size = pop_size
    print(f'PopulationColourRGBTransforms requires_grad={requires_grad}')

    rgb_init_range = (
        self.config['initial_max_rgb'] - self.config['initial_min_rgb'])
    population_reds = (
        np.random.rand(pop_size, num_patches, 1, 1, 1)
        * rgb_init_range) + self.config['initial_min_rgb']
    population_greens = (
        np.random.rand(pop_size, num_patches, 1, 1, 1)
        * rgb_init_range) + self.config['initial_min_rgb']
    population_blues = (
        np.random.rand(pop_size, num_patches, 1, 1, 1)
        * rgb_init_range) + self.config['initial_min_rgb']
    population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
    population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

    self.reds = torch.nn.Parameter(
        torch.tensor(population_reds, dtype=torch.float),
        requires_grad=requires_grad)
    self.greens = torch.nn.Parameter(
        torch.tensor(population_greens, dtype=torch.float),
        requires_grad=requires_grad)
    self.blues = torch.nn.Parameter(
        torch.tensor(population_blues, dtype=torch.float),
        requires_grad=requires_grad)
    self._zeros = torch.nn.Parameter(
        torch.tensor(population_zeros, dtype=torch.float),
        requires_grad=False)
    self.orders = torch.nn.Parameter(
        torch.tensor(population_orders, dtype=torch.float),
        requires_grad=requires_grad)

  def _clamp(self):
    self.reds.data = self.reds.data.clamp(
        min=self.config['min_rgb'], max=self.config['max_rgb'])
    self.greens.data = self.greens.data.clamp(
        min=self.config['min_rgb'], max=self.config['max_rgb'])
    self.blues.data = self.blues.data.clamp(
        min=self.config['min_rgb'], max=self.config['max_rgb'])
    self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

  def copy_and_mutate_s(self, parent, child):
    with torch.no_grad():
      self.reds[child, ...] = (
          self.reds[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.reds[child, ...].shape).to(self.device))
      self.greens[child, ...] = (
          self.greens[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.greens[child, ...].shape).to(self.device))
      self.blues[child, ...] = (
          self.blues[parent, ...]
          + self.config['colour_mutation_scale'] * torch.randn(
              self.blues[child, ...].shape).to(self.device))
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
