import numpy as np
OUTPUT_DIR = "out"

#@title Collage configuration
#@markdown Render methods
#@markdown **opacity** patches overlay each other using a combination of alpha and depth,
#@markdown **transparency** _adds_ patch colours (black therefore appearing transparent),
#@markdown and **masked transparency** blends patches using the alpha channel.
RENDER_METHOD = "transparency"  #@param ["opacity", "transparency", "masked_transparency"]
NUM_PATCHES =      20  #@param {type:"integer"}
COLOUR_TRANSFORMATIONS = "RGB space"  #@param ["none", "RGB space", "HSV space"]
#@markdown Invert image colours to have a white background?
INVERT_COLOURS = False #@param {type:"boolean"}
MULTIPLIER_BIG_IMAGE = 4

#@title Affine transform settings
#@markdown Translation bounds for X and Y.
MIN_TRANS = -0.66  #@param{type:"slider", min:-1.0, max:1.0, step:0.01}
MAX_TRANS = 0.8  #@param{type:"slider", min:-1.0, max:1.0, step:0.01}
#@markdown Scale bounds (> 1 means zoom out and < 1 means zoom in).
MIN_SCALE =   1#@param {type:"number"}
MAX_SCALE =   2#@param {type:"number"
#@markdown Bounds on ratio between X and Y scale (default 1).
MIN_SQUEEZE =   0.5#@param {type:"number"}
MAX_SQUEEZE =   2.0#@param {type:"number"}
#@markdown Shear deformation bounds (default 0)
MIN_SHEAR = -0.2  #@param{type:"slider", min:-1.0, max:1.0, step:0.01}
MAX_SHEAR = 0.2  #@param{type:"slider", min:-1.0, max:1.0, step:0.01}
#@markdown Rotation bounds.
MIN_ROT_DEG = -180 #@param{type:"slider", min:-180, max:180, step:1}
MAX_ROT_DEG = 180 #@param{type:"slider", min:-180, max:180, step:1}
MIN_ROT = MIN_ROT_DEG * np.pi / 180.0
MAX_ROT = MAX_ROT_DEG * np.pi / 180.0

#@title Colour transform settings
#@markdown RGB
MIN_RGB = -0.21  #@param {type:"slider", min: -1, max: 1, step: 0.01}
MAX_RGB = 1.0  #@param {type:"slider", min: 0, max: 1, step: 0.01}
INITIAL_MIN_RGB = 0.05  #@param {type:"slider", min: 0, max: 1, step: 0.01}
INITIAL_MAX_RGB = 0.25  #@param {type:"slider", min: 0, max: 1, step: 0.01}
#@markdown HSV
MIN_HUE = 0.  #@param {type:"slider", min: 0, max: 1, step: 0.01}
MAX_HUE_DEG = 360 #@param {type:"slider", min: 0, max: 360, step: 1}
MAX_HUE = MAX_HUE_DEG * np.pi / 180.0
MIN_SAT = 0.  #@param {type:"slider", min: 0, max: 1, step: 0.01}
MAX_SAT = 1.  #@param {type:"slider", min: 0, max: 1, step: 0.01}
MIN_VAL = 0.  #@param {type:"slider", min: -1, max: 1, step: 0.01}
MAX_VAL = 1.  #@param {type:"slider", min: 0, max: 1, step: 0.01}

#@title Training settings
#@markdown Number of training steps
OPTIM_STEPS = 200    #@param{type:"slider", min:200, max:20000, step:100}
LEARNING_RATE = 0.1    #@param{type:"slider", min:0.0, max:0.6, step:0.01}
#@markdown Number of augmentations to use in evaluation
USE_IMAGE_AUGMENTATIONS = True #@param{type:"boolean"}
NUM_AUGS = 4  #@param {type:"integer"}
#@markdown Normalize colours for CLIP, generally leave this as True
USE_NORMALIZED_CLIP = False  #@param {type:"boolean"}
#@markdown Gradient clipping during optimisation
GRADIENT_CLIPPING = 10.0  #@param {type:"number"}
#@markdown Initial random search size (1 means no search)
INITIAL_SEARCH_SIZE = 1 #@param {type:"slider", min:1, max:50, step:1}

#@title Evolution settings
#@markdown For evolution set POP_SIZE greater than 1
POP_SIZE =    2  #@param{type:"slider", min:1, max:100}
EVOLUTION_FREQUENCY =  100#@param {type:"integer"}
#@markdown **Microbial** - loser of randomly selected pair is replaced by mutated winner. A low selection pressure.
#@markdown **Evolutionary Strategies** - mutantions of the best individual replace the rest of the population. Much higher selection pressure than Microbial GA.
GA_METHOD = "Microbial"  #@param ["Evolutionary Strategies", "Microbial"]
#@markdown ### Mutation levels
#@markdown Scale mutation applied to position and rotation, scale, distortion, colour and patch swaps.
POS_AND_ROT_MUTATION_SCALE = 0.02  #@param{type:"slider", min:0.0, max:0.3, step:0.01}
SCALE_MUTATION_SCALE = 0.02  #@param{type:"slider", min:0.0, max:0.3, step:0.01}
DISTORT_MUTATION_SCALE = 0.02  #@param{type:"slider", min:0.0, max:0.3, step:0.01}
COLOUR_MUTATION_SCALE = 0.02  #@param{type:"slider", min:0.0, max:0.3, step:0.01}
PATCH_MUTATION_PROBABILITY = 1  #@param{type:"slider", min:0.0, max:1.0, step:0.1}
#@markdown Limit the number of individuals shown during training
MAX_MULTIPLE_VISUALISATIONS =   5#@param {type:"integer"}
#@markdown Save video of population sample over time.
POPULATION_VIDEO = True  #@param (type:"boolean")

#@title Load segmented patches
PATCH_SET = "Animals" #@param ["Fruit and veg", "Sea glass", "Animals", "Handwritten MNIST", "Upload to Colab", "Load from URL", "Load from Google Drive"]
URL_TO_PATCH_FILE = "" #@param {type:"string"}

#@title Resize image patches to low- and high-res.
FIXED_SCALE_PATCHES = True #@param {type:"boolean"}
FIXED_SCALE_COEFF =   0.7#@param {type:"number"}
NORMALIZE_PATCH_BRIGHTNESS = False  #@param {type: "boolean"}
PATCH_MAX_PROPORTION =  5  #@param{type:"slider", min:2, max:8, step:1}
PATCH_WIDTH_MIN = 16
PATCH_HEIGHT_MIN = 16

#@markdown Configure a background, e.g. uploaded picture or solid colour.
# NOTE: DO NOT CHANGE THESE STRING TEXTS WITHOUT CHANGING THE RELEVANT CODE!
BACKGROUND = "None (black)" #@param ["None (black)", "Solid colour below", "Upload image to Colab"]
#@markdown Background usage: Global = use image across whole image; Local = reuse same image for every tile
BACKGROUND_USE = "Global" #@param ["Global", "Local"]

#@markdown Colour configuration for solid colour background
BACKGROUND_RED = 195 #@param {type:"slider", min:0, max:255, step:1}
BACKGROUND_GREEN = 181 #@param {type:"slider", min:0, max:255, step:1}
BACKGROUND_BLUE = 172 #@param {type:"slider", min:0, max:255, step:1}

#@markdown URL if downloading image file from website:
BACKGROUND_IMAGE_URL = "" #@param {type:"string"}

# @title Configure image prompt and content
#@markdown Enter a **global** description of the image, e.g. 'a photorealistic chicken'
GLOBAL_PROMPT = "Roman mosaic of an unswept floor"   #@param {type:"string"}

# @title Tile prompts and tiling settings
TILE_IMAGES = True #@param {type:"boolean"}
TILES_WIDE = 1  #@param {type:"slider", min:1, max:10, step:1}
TILES_HIGH = 2  #@param {type:"slider", min:1, max:10, step:1}

#@markdown **Prompt(s) for tiles**
#@markdown **Global tile prompt** uses GLOBAL_PROMPT (previous cell) for *all* tiles (e.g. "Roman mosaic of an unswept floor")
GLOBAL_TILE_PROMPT = False #@param {type:"boolean"}

#@markdown Otherwise, specify multiple tile prompts with columns separated by | and / to delineate new row.
#@markdown E.g. multiple prompts for a 3x2 "landscape" image : "sun | clouds | sky / fields | fields | trees"
TILE_PROMPT_STRING = "hair / face"   #@param {type:"string"}

# Composition prompts
# @title Composition prompts (within tiles)
#@markdown Use additional prompts for different regions
COMPOSITIONAL_IMAGE = False #@param {type:"boolean"}

#@markdown **Single image** (i.e. no tiling) composition prompts
#@markdown Specify 3x3 prompts for each composition region (left to right, starting at the top)
PROMPT_x0_y0 = "a photorealistic sky with sun"   #@param {type:"string"}
PROMPT_x1_y0 = "a photorealistic sky"   #@param {type:"string"}
PROMPT_x2_y0 = "a photorealistic sky with moon"   #@param {type:"string"}
PROMPT_x0_y1 = "a photorealistic tree"   #@param {type:"string"}
PROMPT_x1_y1 = "a photorealistic tree"   #@param {type:"string"}
PROMPT_x2_y1 = "a photorealistic tree"   #@param {type:"string"}
PROMPT_x0_y2 = "a photorealistic field"   #@param {type:"string"}
PROMPT_x1_y2 = "a photorealistic field"   #@param {type:"string"}
PROMPT_x2_y2 = "a photorealistic chicken"   #@param {type:"string"}

#@markdown **Tile** composition prompts
#@markdown This string is formated to autogenerate region prompts from tile prompt. e.g. "close-up of {}"
TILE_PROMPT_FORMATING = "close-up of {}"  #@param {type:"string"}

# Example prompt lists for different settings, where
# PROMPT = "Roman"
# TILE_PROMPT_FORMATING = "close-up of {}"
# TILE_PROMPT_STRING = "sun | clouds | sky / fields | fields | trees"

# 1. Single image with **global** prompt
#   * Tile 0 prompts: ['Roman']
# 1. Single image with **composition** prompts (tested)
#   * Tile 0 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
# 1. Tiled images with **global** prompt for each tile.
#   * Tile 0 prompts: ['Roman']
#   * Tile 1 prompts: ['Roman']
#   * Tile 2 prompts: ['Roman']
#   * Tile 3 prompts: ['Roman']
#   * Tile 4 prompts: ['Roman']
#   * Tile 5 prompts: ['Roman']
# 1. Tiled images with **global** prompt for each tile.
#   * Tile 0 prompts: ['sun']
#   * Tile 1 prompts: ['clouds']
#   * Tile 2 prompts: ['sky']
#   * Tile 3 prompts: ['fields']
#   * Tile 4 prompts: ['fields']
#   * Tile 5 prompts: ['trees']
# 1. Tiled images with separate **composition** prompts for each tile.
#   * Tile 0 prompts: ['close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'close-up of sun', 'sun']
#   * Tile 1 prompts: ['close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'close-up of clouds', 'clouds']
#   * Tile 2 prompts: ['close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'close-up of sky', 'sky']
#   * Tile 3 prompts: ['close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'fields']
#   * Tile 4 prompts: ['close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'close-up of fields', 'fields']
#   * Tile 5 prompts: ['close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'close-up of trees', 'trees']
# [188]
#
# 1. Tiled images with **global** **composition** prompts for each tile.
#   * Tile 0 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
#   * Tile 1 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
#   * Tile 2 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
#   * Tile 3 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
#   * Tile 4 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']
#   * Tile 5 prompts: ['close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'close-up of Roman', 'Roman']


