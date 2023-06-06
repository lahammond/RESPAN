# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'spine-analysis'
__version__   = '0.1.0'
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spine-analysis'


import os
import sys
#import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
import ast

##############################################################################
# Main Functions
##############################################################################
Locations = None
Settings = None

#create dir   
def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

#count dirs
def count_dirs(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            count += 1

    return count

#count files
def count_files(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            count += 1

    return count

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def check_gpu():
  if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.') 

  else:
    print('You have GPU access')
    #!nvidia-smi


#place holder for creating pipeline dirs
def create_pipeline_dirs(Settings, Locations):

    create_dir(Locations.analyzed_images)
    
    if Settings.save_val_data == True:
        create_dir(Locations.validation_dir)

#placeholder for pipeline initialization        
def initialize_QLEAN(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  Create_Locations(data_dir)
    create_QLEAN_dirs(Settings, Locations)
    
    return Settings, Locations
    
class ConfigObject:
    def __init__(self, data):
        self.__dict__.update(data)


 
##############################################################################
# Main Classes
##############################################################################
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Create_Locations():
  def __init__(self, data_dir):
      self.input_dir = data_dir
      self.validation_dir =  data_dir+"Validation_Data/"
      self.analyzed_images = data_dir+"Analyzed_Images/"
      self.tables = data_dir+"Tables/"
      self.plots = data_dir+"Plots/"


#placeholder for creating settings dict      
class Create_Settings():

    def __init__(self, data_dir):
        with open(data_dir+"Analysis_Settings.yaml", 'r') as ymlfile:
            #setting = yaml.safe_load(ymlfile)
            setting = yaml.load(ymlfile, Loader = PrettySafeLoader)

            self.input_resXY = setting["Parameters"]["input_resXY"]
            self.input_resZ = setting["Parameters"]["input_resZ"]

  