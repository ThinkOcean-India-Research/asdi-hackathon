import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import geopandas as gpd
import geoio
import json

# run inside building-detection folder
MRCNN_PATH = 'utils/Mask_RCNN'
sys.path.append(MRCNN_PATH)

# mask-rcnn submodule imports
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


MODEL_DIR = 'model'
COCO_MODEL_PATH = 'model/mask_rcnn_coco.h5'

# download coco base trained weights
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)