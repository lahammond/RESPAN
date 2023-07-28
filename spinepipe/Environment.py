# -*- coding: utf-8 -*-
"""
Environment
===========

Initialize spine analysis environment

Note
----
To initialize the main functions in a spine-analysis script use:
>>> from spine-analysis.Environment import *
"""
__title__     = 'spinpipe'
__version__   = '0.1.0'
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'


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
import spinepipe.Main.Main as main
import spinepipe.Main.Timer as timer


#image processing
import spinepipe.ImageAnalysis.ImageAnalysis as imgan
import spinepipe.ImageAnalysis.Validation as val


#analysis


###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'tifffile', 'time', 'np',
           'plt', 'figure', 'exposure', 
           'segmentation', 'imread', 'imsave', 'imshow',  'util', 'img_as_ubyte',
           'ndi', 'clear_output',
           'display', 'main', 'timer', 'imgan', 'val'];
