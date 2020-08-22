import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dropout,UpSampling2D,concatenate,Activation,Dense,Flatten
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as kb,Input
from PIL import Image, ImageFilter, ImageEnhance


GRID_SIZE = 24
IMAGE_SIZE = 768
IMAGE_SIZE_UNET = 64
IMAGE_FOLDER = "D:\\Data\\Kaggle\\Airbus\\train\\image"
MASK_FOLDER = "D:\\Data\\Kaggle\\Airbus\\train\\mask"
MODEL_SAVE_PATH = "D:\\Data\\Kaggle\\Airbus\\Airbus_FM.R0.hdf5"
