"""Parent class for multimodal evaluators, such as CLIP, ALIGN, etc."""


class MultimodalEncoder():
  """MultimodalEncoder maps multiple modalities to shared embedding space."""

  def __init__(self, config):
    """Init.

    Args:
      config: dictionary of configuration data required for specific model.
    """
    del config
    AssertionError("Missing derived class implemention.")

  def encode_image(self, image_batch):
    """Get embeddings for each image in a batch.

    Args:
      image_batch: batch of images in format for the model.
    """
    del image_batch
    AssertionError("Missing derived class implemention.")

  def encode_text(self, text_list):
    """Get embeddings for each text string in a list of strings.

    Args:
      text_list: list of strings to encode.
    """
    del text_list
    AssertionError("Missing derived class implemention.")
