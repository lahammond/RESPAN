# -*- coding: utf-8 -*-
"""
Batch Processing Script
==========


"""
__title__     = 'SpinePipe'
__version__   = '0.9.7'
__date__      = "2 February, 2024"
__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'

"""
SpinePipe Batch Processing

 - Expects 3D Tif files acquired using parameters matched to the nnUnet model selected for inference
 - multichannel tifs accepted
 - Dendrite and spines segmented, with subsquent visual and tabular readouts
 - spines can be isolated for further visualization, analysis, or for training ML for morphology detection
 
 Additional Features:
 - Measurement of intensity in other channels 
 - neuron branch analysis - and linked with spine numbers and moprhology
 - 
"""


#spinepipe install directory to path
import sys
import os
import logging
from datetime import datetime

sys.path.append('D:/Dropbox/Github/spine-analysis')

#import SpinePipe modules
from spinepipe.Environment import *

main.check_gpu()




#%%

#Dataset Directories

data_dirs = ["D:/Project_Data/SpinePipe/2024_Baptiste_Data/"]

#data_dirs = ["D:/Project_Data/SpinePipe/Testing/Spine_tracking/"]
 #            ] #,
            # "D:/Project_Data/SpineAnalysis/Testing/FastTest2"]
            #"D:/Project_Data/SpineAnalysis/1_Spine_Tracking_Datasets/Animal1/Segment1/"

for data_dir in data_dirs:
    data_dir = data_dir +"/"    
    if os.path.isdir(data_dir):
        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path): 
                subfolder_path = subfolder_path +"/"    
                print(f"Processing subfolder: {subfolder_path}")
             
            
                # Create a logger
                logger = logging.getLogger(__name__)
                logger.setLevel(logging.DEBUG)
                # create a file handler
                logname = subfolder_path+'SpinePipe_Log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'.log'
                handler = logging.FileHandler(logname)
                handler.setLevel(logging.DEBUG)
                
                # create a logging format
                #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                formatter = logging.Formatter('%(message)s')
                handler.setFormatter(formatter)
                
                    # create a stream handler for stdout
                ch = logging.StreamHandler(sys.stdout)
                ch.setFormatter(formatter)
                
                # add the handlers to the logger
                logger.addHandler(handler)
                logger.addHandler(ch)
                
                logger.info("SpinePipe Version: "+__version__)
                logger.info("Release Date: "+__date__) 
                logger.info("Created by: "+__author__+"")
                logger.info("Department of Neurology, The Ohio State University\nZuckerman Institute, Columbia University\n")    
              
                #Load in experiment parameters and analysis settings   
                settings, locations = main.initialize_spinepipe(subfolder_path)
                 #%%
                #Modify specific parameters and settings:    
                settings.save_intermediate_data = True
                settings.roi_volume_size = 2
                settings.spine_roi_volume_size = 1 #4 in microns in x, y, z - approx 13px for 0.3 resolution data
                settings.GPU_block_size = (150,500,500) #dims used for processing images in block for cell extraction. Reduce if recieving out of memory errors
                settings.neuron_spine_size = (0.03,15)
                settings.neuron_spine_dist = 3
                settings.min_dendrite_vol = 0.5
                
                settings.min_dendrite_vol = round(settings.min_dendrite_vol / settings.input_resXY/settings.input_resXY/settings.input_resZ, 0)
                settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in settings.neuron_spine_size] 
                settings.neuron_spine_dist = round(settings.neuron_spine_dist / (settings.input_resXY),2)
              
                
                
                settings.HistMatch = False
                settings.Track = False
                settings.reg_method = "Elastic" #or "Elastic"
                settings.erode_shape = (0.5,2,2)
                
                settings.image_restore = False
                settings.neuron_channel = 1
                #settings.seg_model_path = "D:/Dropbox/Github/spine-analysis/spinepipe/Models/Dataset022_SpinesDendrites/"
                settings.neuron_seg_model_path = "D:/nnUnet/results/Dataset816_SpinningDisk_SpinesDendritesSoma_Isotropic/"
                #settings.seg_model = ("nnUnet","")
                settings.Vaa3d = True
                settings.analysis_method =  "Whole Neuron" # "Dendrite Specific", or "Whole Neuron" 
                
            
                logger.info("Processing folder: "+subfolder_path)
                logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
                #logger.info(f" Model used: {settings.neuron_seg_model_path}")    
                #logger.info(f" Model resolution: {settings.neuron_seg_model_res[0]}um XY, {settings.neuron_seg_model_res[2]}um Z")    
                logger.info(f" Spine volume set to: {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
                logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}") 
            
                #Processing
            
                log = imgan.restore_and_segment(settings, locations, logger)
                
                imgan.analyze_spines(settings, locations, log, logger)
                
                if settings.Track == True:
                    strk.track_spines(settings, locations, log, logger)
                    imgan.analyze_spines_4D(settings, locations, log, logger)
                
                
                #Finish log
                
                logger.info("SpinePipe analysis complete.")
                logger.info("\nSpinePipe Version: "+__version__)
                logger.info("Release Date: "+__date__) 
                logger.info("Created by: "+__author__+"")    
                logger.info("Department of Neurology, The Ohio State University\nZuckerman Institute, Columbia University\n")    
                
                logger.info("--------------------------------------------------------------------")
                
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)


        
        
    


