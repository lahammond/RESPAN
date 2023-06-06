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
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__webpage__   = 'http://cellularimaging.org'
__download__  = 'http://www.github.com/lahmmond/spine-analysis'

###############################################################################
### Python
###############################################################################


#clean up libraries
import sys   
import os    

import tifffile
import pims
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
import spineanalyis.Main.Main as main


#image processing
import spineanalysis.ImageAnalysis.ImageAnalysis as imgan


#analysis


###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'tifffile', 'pims', 'time', 'np',
           'plt', 'figure', 'exposure', 
           'segmentation', 'imread', 'imsave', 'imshow',  'util', 'img_as_ubyte',
           'ndi', 'clear_output',
           'display', 'main', 'imgan'];
