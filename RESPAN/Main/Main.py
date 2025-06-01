# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'RESPAN'
__version__   = '0.9.0'
__date__      = "25 July, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


import os
import sys
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

    #create_dir(Locations.plots)
    create_dir(Locations.validation_dir)
    create_dir(Locations.labels)
    create_dir(Locations.restored)
    create_dir(Locations.nnUnet_input)
    create_dir(Locations.arrays)
    create_dir(Locations.Meshes)
    #create_dir(Locations.nnUnet_2nd_pass)

    create_dir(Locations.MIPs)
    create_dir(Locations.Vols)

    
    
def initialize_RESPAN(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  Create_Locations(data_dir)
    create_dirs(Settings, Locations)
    
    return Settings, Locations

def initialize_RESPAN_validation(data_dir):
    
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
        self.Vols =  data_dir+"Validation_Data/Validation_Vols/"
        self.Meshes = data_dir+"Validation_Data/Spine_Meshes/"
        self.tables = data_dir+"Tables/"
        self.plots = data_dir+"Plots/"
        self.arrays = data_dir+"Spine_Arrays/"
        self.nnUnet_input = data_dir+"nnUnet_input/"
        self.nnUnet_2nd_pass = data_dir+"nnUnet_2nd_pass/"
        self.swcs = data_dir+"SWC_files/"
      
    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")


class Create_Settings():

    def __init__(self, data_dir):
        settings_file = os.path.join(str(data_dir), "Analysis_Settings.yaml")
        with open(settings_file, 'r') as ymlfile:
            #setting = yaml.safe_load(ymlfile)
            setting = yaml.load(ymlfile, Loader = PrettySafeLoader)
            
            self.input_resXY = setting["Parameters"]["input_resXY"]
            self.input_resZ = setting["Parameters"]["input_resZ"]
            self.model_resXY = setting["Parameters"]["model_resXY"]
            self.model_resZ = setting["Parameters"]["model_resZ"]
            
            self.tiles_for_prediction = list(map(int, (ast.literal_eval(setting["Analysis"]["tiles_for_prediction"]))))
            self.roi_volume_size = setting["Analysis"]["roi_volume_size"]            
            self.GPU_block_size = list(map(int, (ast.literal_eval(setting["Analysis"]["GPU_block_size"]))))
            self.erode_shape = list(map(int, (ast.literal_eval(setting["Analysis"]["erode_shape"]))))
            self.remove_touching_boarders = setting["Analysis"]["remove_touching_boarders"]
            
            self.validation_format = setting["Analysis"]["validation_format"]
            self.validation_scale = list(map(int, (ast.literal_eval(setting["Analysis"]["validation_scale"]))))
            self.validation_jpeg_comp = setting["Analysis"]["validation_jpeg_comp"]
            try:
                self.additional_logging = setting["Analysis"]["additional_logging"]
            except:
                self.additional_logging = False
  
            self.c1_restore = setting["Channel1"]["restore"]
            self.c1_rest_model_path = setting["Channel1"]["rest_model_path"]
            self.c1_rest_type = ast.literal_eval(setting["Channel1"]["rest_type"])
            
            self.c2_restore = setting["Channel2"]["restore"]
            self.c2_rest_model_path = setting["Channel2"]["rest_model_path"]
            self.c2_rest_type = ast.literal_eval(setting["Channel2"]["rest_type"])
            
            self.c3_restore = setting["Channel3"]["restore"]
            self.c3_rest_model_path = setting["Channel3"]["rest_model_path"]
            self.c3_rest_type = ast.literal_eval(setting["Channel3"]["rest_type"])
            
            self.c4_restore = setting["Channel4"]["restore"]
            self.c4_rest_model_path = setting["Channel4"]["rest_model_path"]
            self.c4_rest_type = ast.literal_eval(setting["Channel4"]["rest_type"])

            #skip if not found in the file in case older setting file
            try:
                self.selfnet_path = setting["SelfNet"]["model_path"]
            except:
                self.selfnet_path = None

            # skip if not found in the file in case older setting file
            try:
                self.refinement_model_path = setting["Refinement"]["model_path"]
                self.refinement_Z = setting["Refinement"]["model_resZ"]
                self.refinement_XY = setting["Refinement"]["model_resXY"]
            except:
                self.refinement_model_path = None
                self.refinement_Z = None
                self.refinement_XY = None

            self.Vaa3Dpath = setting["Vaa3D"]["path"]

    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")

