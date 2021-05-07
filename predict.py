
import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd
import glob
sys.path.append('Mask_RCNN')
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
os.environ['DISPLAY'] = ':0'

# os.chdir('Mask_RCNN')
# To find local version of the library
from keras import backend as K
K.clear_session()

def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 100


class InferenceConfig(DetectorConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_lungs():

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir="Mask_RCNN")

    model.load_weights('mask_rcnn_pneumonia_0029.h5', by_name=True)

    original_image = cv2.imread('image.png')
    results = model.detect([original_image])
    K.clear_session()
    r = results[0]
    k = len(r['scores'][r['scores'] > 0.9])
    visualize.display_instances(original_image, r['rois'][:k], r['masks'][:, :, :k], r['class_ids'][:k],
                                ['BG', 'Lung Opacity'], r['scores'][:k],
                                colors=get_colors_for_class_ids(r['class_ids'][:k]))  # else:
    img = cv2.imread('predicted_image.png')
