# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'spinpipe'
__version__   = '0.9.0'
__date__      = "25 July, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'


import os
import sys
#import numpy as np
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
  """
    if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.') 

  else:
    print('You have GPU access')
    #!nvidia-smi
"""

#place holder for creating pipeline dirs
def create_dirs(Settings, Locations):

    create_dir(Locations.tables)
    create_dir(Locations.plots)
    create_dir(Locations.validation_dir)
    create_dir(Locations.labels)
    create_dir(Locations.restored)
    create_dir(Locations.nnUnet_input)
    create_dir(Locations.arrays)
        
    if Settings.save_val_data == True:
        create_dir(Locations.MIPs)
    
def initialize_spinepipe(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  Create_Locations(data_dir)
    create_dirs(Settings, Locations)
    
    return Settings, Locations

def initialize_spinepipe_validation(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  data_dir
    
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
        self.labels =  data_dir+"Validation_Data/Segmentation_Labels/"
        self.restored =  data_dir+"Validation_Data/Restored_Images/"
        self.MIPs =  data_dir+"Validation_Data/Validation_MIPs/"
        self.tables = data_dir+"Tables/"
        self.plots = data_dir+"Plots/"
        self.arrays = data_dir+"Spine_Arrays/"
        self.nnUnet_input = data_dir+"nnUnet_input/"
      
    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")


class Create_Settings():

    def __init__(self, data_dir):
        with open(data_dir+"Analysis_Settings.yaml", 'r') as ymlfile:
            #setting = yaml.safe_load(ymlfile)
            setting = yaml.load(ymlfile, Loader = PrettySafeLoader)

            self.input_resXY = setting["Parameters"]["input_resXY"]
            self.input_resZ = setting["Parameters"]["input_resZ"]
            
            self.tiles_for_prediction = list(map(int, (ast.literal_eval(setting["Analysis"]["tiles_for_prediction"]))))
            self.roi_volume_size = setting["Analysis"]["roi_volume_size"]            
            self.GPU_block_size = list(map(int, (ast.literal_eval(setting["Analysis"]["GPU_block_size"]))))
            
            self.spine_roi_volume_size = setting["Analysis"]["spine_roi_volume_size"]
            self.erode_shape = list(map(int, (ast.literal_eval(setting["Analysis"]["erode_shape"]))))
            self.remove_touching_boarders = setting["Analysis"]["remove_touching_boarders"]
            
            self.save_intermediate_data = setting["Analysis"]["save_intermediate_data"]
            self.save_val_data = setting["Analysis"]["save_val_data"]
            self.validation_format = setting["Analysis"]["validation_format"]
            self.validation_scale = list(map(int, (ast.literal_eval(setting["Analysis"]["validation_scale"]))))
            self.validation_jpeg_comp = setting["Analysis"]["validation_jpeg_comp"]
  
            self.neuron_channel = setting["Neuron"]["channel"]
            self.neuron_restore = setting["Neuron"]["restore"]
            self.neuron_rest_model_path = setting["Neuron"]["rest_model_path"]
            self.neuron_rest_type = ast.literal_eval(setting["Neuron"]["rest_type"])
            self.neuron_seg_model_path = setting["Neuron"]["seg_model_path"]
            self.neuron_seg_model = ast.literal_eval(setting["Neuron"]["seg_model"])
            self.neuron_seg_step = list(map(int, (ast.literal_eval(setting["Neuron"]["seg_model_step"]))))
            self.neuron_seg_model_res = list(map(float, (ast.literal_eval(setting["Neuron"]["seg_model_res"]))))
            self.neuron_prob_thresh = setting["Neuron"]["prob_thresh"]
            self.neuron_spine_size = list(map(int, (ast.literal_eval(setting["Neuron"]["spine_size"]))))
            self.neuron_intensity_filter = setting["Neuron"]["intensity_filter"]

            self.nnUnet_raw = setting["nnUnet"]["raw"]
            self.nnUnet_preprocessed = setting["nnUnet"]["preprocessed"]
            self.nnUnet_results = setting["nnUnet"]["results"]
            self.nnUnet_type = setting["nnUnet"]["type"]
            self.nnUnet_conda_path = setting["nnUnet"]["conda_path"]
            self.nnUnet_env = setting["nnUnet"]["env"]
    
    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")

