"""CLIP encoder for use in Arnheim 3."""
from . import multimodal_encoder
import clip
import torch


class CLIPEncoder(multimodal_encoder.MultimodalEncoder):
  """CLIP encoder through MultimodalEncoder interface."""

  def __init__(self, config, device=None):
    """Init.

    Args:
      config: dictionary of configuration data required for specific model.
    """
    if device is None:
      raise ValueError("CLIPEncoder requires a device arg. None given.")
    self._device = device
    # Initialise and load CLIP model.
    print(f"Downloading CLIP model {config['clip_model']}...")
    self._clip_model, _ = clip.load(
        config["clip_model"], self._device, jit=False)

  def encode_images(self, image_batch):
    """Get embeddings for each image in a batch.

    Args:
      image_batch: batch of images in format for the model.
    """
    return self._clip_model.encode_image(image_batch)

  def encode_texts(self, text_list):
    """Get embeddings for each text string in a list of strings.

    Args:
      text_list: list of strings to encode.
    """

    text_inputs = []
    for prompt in text_list:
      text_inputs.append(clip.tokenize(prompt).to(self._device))

    features = []
    with torch.no_grad():
      for text_input in text_inputs:
        features.append(self._clip_model.encode_text(text_input))
    return features
