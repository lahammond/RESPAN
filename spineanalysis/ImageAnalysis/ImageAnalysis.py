# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/QLEAN'




import os
import numpy as np
import pandas as pd
import ast

import math

import spineanalysis.Main.Main as main

import segmentation_models as sm
from keras.models import load_model
from patchify import patchify, unpatchify

import subprocess
from subprocess import Popen, PIPE

#from tqdm import tqdm

import cupy as cp
import cupyx.scipy.ndimage as cupy_ndimage


import cv2
import tifffile
from skimage.io import imread, imsave, imshow, util

from math import trunc

import skimage.io
from skimage.transform import rescale
from skimage import color, data, filters, measure, morphology, segmentation, util, exposure, restoration
from skimage.measure import label
from scipy import ndimage

import gc
#import cupy as cp

#import matplotlib.pyplot as plt

from csbdeep import data

from stardist.models import StarDist2D
import tensorflow as tf
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict, plot_history, Path, download_and_extract_zip_file, normalize
from csbdeep.data import RawData, create_patches 
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE

import QLEAN.ImageAnalysis.unet as unet

##############################################################################
# Main Processing Functions
##############################################################################
