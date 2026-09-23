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

try:
    from IPython.display import clear_output, display
except ImportError:
    # IPython is a notebook convenience re-exported for interactive use; it is
    # not a pipeline dependency and is absent from the packaged application
    # and the CPU test environments. Provide inert stand-ins so importing this
    # module (which the GUI's Run button does) never fails on its account.
    def clear_output(*args, **kwargs):
        pass

    def display(*args, **kwargs):
        pass



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
import RESPAN.ImageAnalysis.Segmentation_and_Restoration as sr
import RESPAN.ImageAnalysis.IO as io


#analysis


###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'tifffile', 'time', 'np',
           'plt', 'figure', 'exposure', 
           'segmentation', 'imread', 'imsave', 'imshow',  'util', 'img_as_ubyte',
           'ndi', 'clear_output',
           'display', 'main', 'timer', 'imgan', 'val', 'strk', 'mt', 'sr', 'io',];
