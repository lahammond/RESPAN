# -*- coding: utf-8 -*-
"""
Environment
===========

Initialize spine analysis environment

Note
----
To initialize the main functions in a spine-analysis script use:
>>> from RESPAN.Environment import *
"""
__title__     = 'RESPAN'
__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'MIT License (see LICENSE)'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


###############################################################################
### Python
###############################################################################


#clean up libraries
import sys   
import os    

import tifffile
#import pims
import time

import numpy as np                
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from skimage import exposure, segmentation
from skimage.io import imread, imsave, imshow, util
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi #Distance transformation

from IPython.display import clear_output, display



###############################################################################
### QLEAN
###############################################################################

#Utilities

#Main
import RESPAN.Main.Main as main
import RESPAN.Main.Timer as timer


#image processing
import RESPAN.ImageAnalysis.ImageAnalysis as imgan
import RESPAN.ImageAnalysis.Validation as val
import RESPAN.ImageAnalysis.SpineTracking as strk
import RESPAN.ImageAnalysis.ModelTraining as mt


#analysis


###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'tifffile', 'time', 'np',
           'plt', 'figure', 'exposure', 
           'segmentation', 'imread', 'imsave', 'imshow',  'util', 'img_as_ubyte',
           'ndi', 'clear_output',
           'display', 'main', 'timer', 'imgan', 'val', 'strk', 'mt'];
