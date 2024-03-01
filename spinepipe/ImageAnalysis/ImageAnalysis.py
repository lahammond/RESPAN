# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""
__title__     = 'SpinePipe'
__version__   = '0.9.7'
__date__      = "2 February, 2024"
__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'

import os
import numpy as np
import pandas as pd
#import ast

import math

#import time
import warnings
import re
import shutil
from pathlib import Path

import spinepipe.Main.Main as main
#import spinepipe.Main.Timer as timer


import subprocess
#from subprocess import Popen, PIPE

from tqdm import tqdm

import cupy as cp
#import cupyx.scipy.ndimage as cupy_ndimage


#import cv2
import tifffile
#from skimage.io import imread #, util #imsave, imshow,
from tifffile import imread

#from math import trunc

#import skimage.io
#from skimage.transform import rescale
from skimage import measure, morphology, segmentation, exposure #util,  color, data, filters,  exposure, restoration 
from skimage.transform import resize
#from skimage.measure import label
from scipy import ndimage

from csbdeep.models import CARE

import sys
import gc

import contextlib


#####
# to prevent tqdm progress bar conflicting with GUI (CARE predict function issue)

@contextlib.contextmanager
def suppress_all_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
##############################################################################
# Main Processing Functions
##############################################################################

def restore_and_segment(settings, locations, logger):
    
    if settings.image_restore == True:
        restore_image(locations.input_dir, settings, locations, logger)
        #include options here for alternative unets if required
        log = nnunet_create_labels(locations.restored, settings, locations, logger)
    else:
        #include options here for alternative unets if required
        log = nnunet_create_labels(locations.input_dir, settings, locations, logger)
        
    return log
    
    
def restore_image (inputdir, settings, locations, logger):    
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Restoring neuron images...")
        
    files = [file_i for file_i in os.listdir(inputdir) if file_i.endswith('.tif')]
    files = sorted(files)

    

    for file in range(len(files)):
        logger.info(f' Restoring image {files[file]} ')

        image = imread(inputdir + files[file])
        logger.info(f"  Raw data has shape {image.shape}")

        image = check_image_shape(image, logger)
        
        restored = np.empty((image.shape[0], image.shape[1], image.shape[2], image.shape[3]), dtype=np.uint16)
        
        for channel in range(image.shape[1]):
            restore_on = getattr(settings, f'c{channel+1}_restore', None)
            rest_model_path = getattr(settings, f'c{channel+1}_rest_model_path', None)
            rest_type = getattr(settings, f'c{channel+1}_rest_type', None)
            if restore_on == True and rest_model_path != None:
                logger.info(f"  Restoring channel {channel+1}")
                logger.info(f"  Restoration model = {rest_model_path}\n  ---")
                
                channel_image = image[:,channel,:,:]
                  
        
                if rest_type[0] == 'care':
                    if os.path.isdir(rest_model_path) is False:
                        raise RuntimeError(rest_model_path, "not found, check settings and model directory")
                    rest_model = CARE(config=None, name=rest_model_path)
                    
                    #restored = np.empty((channel_image.shape[0], channel_image.shape[1], channel_image.shape[2]), dtype=np.uint16)

                    with suppress_all_output(), main.HiddenPrints():
       
                        #restore image
                        restored_channel = rest_model.predict(channel_image, axes='ZYX', n_tiles=settings.tiles_for_prediction)
        
                        #convert to 16bit
                        restored_channel= restored_channel.astype(np.uint16)

                        restored[:,channel, :, :] = restored_channel
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if settings.validation_format == "tif":
                tifffile.imwrite(locations.restored+ files[file], restored, imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))
                        
    logger.info("Restoration complete.\n\n-----------------------------------------------------------------------------------------------------")
def check_image_shape(image,logger):
    if len(image.shape) > 3:
        #Multichannel input format variability
        # Enable/modify if issues with different datasets to ensure consistency
        smallest_axis = np.argmin(image.shape)
        if smallest_axis != 1:
            # Move the smallest axis to position 2
            image = np.moveaxis(image, smallest_axis, 1)
            logger.info(f"Channels moved ZCYX - raw data now has shape {image.shape}") #ImageJ supports TZCYX order 
    else:
        image = np.expand_dims(image, axis=1)
            
    return image
            

class FakeStream(object):
    def isatty(self):
        return False



def nnunet_create_labels(inputdir, settings, locations, logger):
    logger.info(f"Detecting spines and dendrites...")
    settings.shape_error = False
    settings.rescale_req = False
    
    #check if rescaling required and create scaling factors
    if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
        logger.info(f" Images will be rescaled to match network.")
        
        settings.rescale_req = True
        #z in / z desired, y in / desired ...
        settings.scaling_factors = (settings.input_resZ/settings.model_resZ,
                                    settings.input_resXY/settings.model_resXY,
                                    settings.input_resXY/settings.model_resXY)
        #settings.inverse_scaling_factors = tuple(1/np.array(settings.scaling_factors))
        
        logger.info(f" Scaling factors: Z = {settings.scaling_factors[0]} Y = {settings.scaling_factors[1]} X = {settings.scaling_factors[2]} ")
    
    #data can be raw data OR restored data so check channels
    
    files = [file_i
             for file_i in os.listdir(locations.input_dir)
             if file_i.endswith('.tif')]
    files = sorted(files)

    label_files = [file_i
                   for file_i in os.listdir(locations.labels)
                   if file_i.endswith('.tif')]
    
    #create empty arrays to capture dims and padding info
    settings.original_shape = [None] * len(files)
    settings.padding_req = np.zeros(len(files))
    
    if len(files) == len(label_files):
        logger.info(f" *Spines and dendrites already detected. \nDelete \Validation_Data\Segmentation_Labels if you wish to regenerate.")
        stdout = None
        settings.prev_labels = True
        
    
    else:
        
        #Prepare Raw data for nnUnet
        
        # Initialize reference to None - if using histogram matching
        logger.info(f" Histogram Matching is set to = {settings.HistMatch}")
        reference_image = None
        settings.prev_labels = False
       
        for file in range(len(files)):
            logger.info(f" Preparing image {file+1} of {len(files)} - {files[file]}")
            
            image = imread(inputdir + files[file])
            logger.info(f"  Raw data has shape: {image.shape}")
            
            image = check_image_shape(image, logger)
            
            neuron = image[:,settings.neuron_channel-1,:,:]
            
    
            
            # rescale if required by model        
            if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                settings.original_shape[file] = neuron.shape
                #new_shape = (int(neuron.shape[0] * settings.scaling_factors[0]), neuron.shape[1] * settings.scaling_factors[1]), neuron.shape[2] * settings.scaling_factors[2]))
                new_shape = tuple(int(dim * factor) for dim, factor in zip(neuron.shape, settings.scaling_factors))
                neuron = resize(neuron, new_shape, mode='constant', preserve_range=True, anti_aliasing=True)
                logger.info(f"  Data rescaled for labeling has shape: {neuron.shape}")
    
    
            if neuron.shape[0] < 5:
                #settings.shape_error = True
                #logger.info(f"  !! Insufficient Z slices - please ensure 5 or more slices before processing.")
                #logger.info(f"  File has been moved to \\Not_processed and excluded from processing.")
                #if not os.path.isdir(inputdir+"Not_processed/"):
                #    os.makedirs(inputdir+"Not_processed/")
                #os.rename(inputdir+files[file], inputdir+"Not_processed/"+files[file])
                
                
                #Padding and flag this file as padded for unpadding later
                # Pad the array
                settings.padding_req[file] = 1
                neuron = np.pad(neuron, pad_width=((2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0) 
                logger.info(f"  Too few Z-slices, padding to allow analysis.")
                    
            if settings.HistMatch == True:
                # If reference_image is None, it's the first image.
                if reference_image is None:
                    #neuron = contrast_stretch(neuron, pmin=0, pmax=100)
                    reference_image = neuron
                else:
                   # Match histogram of current image to the reference image
                   neuron = exposure.match_histograms(neuron, reference_image)    
                
            #logger.info(f" ")
            #save neuron as a tif file in nnUnet_input - if file doesn't end with 0000 add that at the end
            name, ext = os.path.splitext(files[file])
    
            if not name.endswith("0000"):
                name += "_0000"
    
            new_filename = name + ext
            
            filepath = locations.nnUnet_input+new_filename
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tifffile.imwrite(filepath, neuron.astype(np.uint16), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))
    
                
        #Run nnUnet over prepared files
        #initialize_nnUnet(settings)
        
        # split the path into subdirectories
        subdirectories = os.path.normpath(settings.neuron_seg_model_path).split(os.sep)
        last_subdirectory = subdirectories[-1]
        # find all three digit sequences in the last subdirectory
        matches = re.findall(r'\d{3}', last_subdirectory)
        # If there's a match, assign it to a variable
        dataset_id = matches[0] if matches else None
    
        
        logger.info("\nPerforming spine and dendrite detection on GPU...")
        
        ##uncomment if issues with nnUnet
        #logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")
        
        stdout, cmd = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, 
                                    locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, settings, logger)
        
        #logger.info(cmd)
        
        ##uncomment if issues with nnUnet
        #result = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, locations,settings)
        
        '''
        # Add environment to the system path
        #os.environ["PATH"] = settings.nnUnet_env_path + os.pathsep + os.environ["PATH"]
    
        #python = settings.nnUnet_env_path+"/python.exe"
    
        #result = subprocess.run(['nnUNetv2_predict', '-i', locations.nnUnet_input, '-o', locations.labels, '-d', dataset_id, '-c', settings.nnUnet_type, '-f', 'all'], capture_output=True, text=True)
        #command = 'nnUNetv2_predict -i ' + locations.nnUnet_input + ' -o ' + locations.labels + ' -d ' + dataset_id + ' -c ' + settings.nnUnet_type + ' -f all']
        
        #command = [python, "-m", "nnUNetv2_predict", "-i", locations.nnUnet_input, "-o", locations.labels, "-d", dataset_id, "-c", settings.nnUnet_type, "-f", "all"]
        #result = run_command_in_conda_env(command, settings.env ,settings.python)
        #result = subprocess.run(command, capture_output=True, text=True)
    
        #logger.info(result.stdout)  # This is the standard output of the command.
        #logger.info(result.stderr)  # This is the error output of the command. 
        '''
        #logger.info(stdout)
        
        #delete nnunet input folder and files
        if settings.save_intermediate_data == False and settings.Track == False:
            if os.path.exists(locations.nnUnet_input):
                shutil.rmtree(locations.nnUnet_input)
        
        #Clean up label folder
    
        if os.path.exists(locations.labels):
            # iterate over all files in the directory
            for filename in os.listdir(locations.labels):
                # check if the file is not a .tif file
                if not filename.endswith('.tif'):
                    # construct full file path
                    file_path = os.path.join(locations.labels, filename)
                    # remove the file
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        

        
        #if tracking over time then we want unpad and match how the labels will appear
        if settings.Track == True:
            
            files = [file_i
                     for file_i in os.listdir(locations.nnUnet_input)
                     if file_i.endswith('.tif')]
            files = sorted(files)
            
            for file in range(len(files)):
                if file == 0: logger.info(' Unpadding and rescaling neuron channel for registration and time tracking...')
                
                #Unpad if padded
                if settings.padding_req[file] == 1:
                    image = image[2:-2, :, :]
                image = imread(locations.nnUnet_input + files[file])
          
                
                
                
                # rescale labels back up if required      
                if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                    #logger.info(f"orignal settings shape: {settings.original_shape[file]}")
                    image = resize(image, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                    #logger.info(f"image resized: {labels.shape}")
        
                    tifffile.imwrite(locations.nnUnet_input + files[file], image.astype(np.uint8), imagej=True, photometric='minisblack',
                            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                            resolution=(settings.input_resXY, settings.input_resXY))
                    
        
    #logger.info("Segmentation complete.\n")
    logger.info("\n-----------------------------------------------------------------------------------------------------")
    return stdout

def run_nnunet_predict(conda_dir, nnUnet_env, input_dir, output_dir, dataset_id, nnunet_type, settings, logger):
    # Set environment variables
    
    initialize_nnUnet(settings, logger)
    
    activate_env = fr"{conda_dir}\Scripts\activate.bat && set PATH={conda_dir}\envs\{nnUnet_env}\Scripts;%PATH%"

    # Define the command to be run
    cmd = "nnUNetv2_predict -i \"{}\" -o \"{}\" -d {} -c {} -f all".format(input_dir, output_dir, dataset_id, nnunet_type)
    
       
    # Combine the activate environment command with the actual command
    final_cmd = f'{activate_env} && {cmd}'

    # Run the command
    process = subprocess.Popen(final_cmd, shell=True)
    stdout, stderr = process.communicate()
    return stdout, cmd
    
def run_nnunet_predict_bat(path, env, input_dir, output_dir, dataset_id, nnunet_type,locations, settings):
    # Define the batch file content
    batch_file_content = f"""@echo off
    CALL {path}\\Scripts\\activate.bat\nconda activate {env}
    set nnUNet_raw={settings.nnUnet_raw}
    set nnUNet_preprocessed={settings.nnUnet_preprocessed}
    set nnUNet_results={settings.nnUnet_results}
    nnUNetv2_predict -i "{input_dir}" -o "{output_dir}" -d {dataset_id} -c {nnunet_type} -f all"""
    
    # Define the batch file path
    batch_file_path = locations.input_dir+"run_nnunet.bat"

    # Write the content to the batch file
    with open(batch_file_path, "w") as batch_file:
        batch_file.write(batch_file_content)

    # Define the command to execute the batch file
    command = [batch_file_path]

    # Execute the batch file
    result = subprocess.run(command, capture_output=True, text=True)

    # Delete the batch file
    #os.remove(batch_file_path)

    return result

def run_command_in_conda_env(command, env, python):
    #command = f"{python} activate {env} && {command}"
    #
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result

def initialize_nnUnet(settings, logger):
    #not worrying about setting raw and processed, as not needed and would rquire additional params for user/settings file
    #os.environ['nnUNet_raw'] = settings.nnUnet_raw
    #os.environ['nnUNet_preprocessed'] = settings.nnUnet_preprocessed
    nnUnet_results = Path(settings.neuron_seg_model_path).parent
    nnUnet_results = str(nnUnet_results).replace("\\", "/")
    os.environ['nnUNet_results'] = nnUnet_results



def analyze_spines(settings, locations, log, logger):
    logger.info("Analyzing spines...")
    #spines = 1
    #dendrites = 2
    #soma = 3
    
    files = [file_i
             for file_i in os.listdir(locations.input_dir)
             if file_i.endswith('.tif')]
    files = sorted(files)

    label_files = [file_i
                   for file_i in os.listdir(locations.labels)
                   if file_i.endswith('.tif')]

    label_files = sorted(label_files)

    if len(files) != len(label_files):
        logger.info(log)
        raise RuntimeError("Number of raw and label images are not the same - check data.")
    
    spine_summary = pd.DataFrame()
    
    for file in range(len(files)):
        logger.info(f' Analyzing image {file+1} of {len(files)} \n  Raw Image: {files[file]} & Label Image: {label_files[file]}')
        
        image = imread(locations.input_dir + files[file])
        labels = imread(locations.labels + files[file]) # use original file name to ensure correct image regardless of sorting
        
        logger.info(f"  Raw shape: {image.shape} & Labels shape: {labels.shape}")
        
        #Unpad if padded # later update - these can be included in Unet processing stage to simplify!
        if settings.padding_req[file] == 1 and settings.prev_labels == False:
            labels = labels[2:-2, :, :]

            tifffile.imwrite(locations.labels + label_files[file], labels.astype(np.uint8), imagej=True, photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                    resolution=(settings.input_resXY, settings.input_resXY))
        
        
        # rescale labels back up if required      
        if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY and settings.prev_labels == False:
            logger.info(f"  orignal settings shape: {settings.original_shape[file]}")
            labels = resize(labels, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
            logger.info(f"  labels resized: {labels.shape}")

            tifffile.imwrite(locations.labels + label_files[file], labels.astype(np.uint8), imagej=True, photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                    resolution=(settings.input_resXY, settings.input_resXY))
            

        #process images through Vaa3d
        if settings.Vaa3d == True and os.path.exists(settings.Vaa3Dpath):
            logger.info(f"  Creating SWC file using Vaa3D...")
            if os.path.exists(locations.swcs + files[file]+".swc"):
                logger.info(f"   {files[file]}.swc already exists, delete this file to regenerate.")
            else:
                vaa3D_neuron = (labels >= 2)*255
                tifffile.imwrite(locations.swcs + files[file], vaa3D_neuron.astype(np.uint8), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
                #run Vaa3D on image:
                cmd = '"{}" /x vn2 /f app2 /i "{}" /o "{}" /p NULL 0 1 1 1 1 0 5 1 0 0'.format(settings.Vaa3Dpath, locations.swcs + files[file], locations.swcs + files[file]+".swc ")
                #logger.info(cmd)
                # Run the command
                process = subprocess.Popen(cmd, shell=True)
                stdout, stderr = process.communicate()
                #logger.info(stdout + stderr)
                
                os.remove(locations.swcs + files[file])
                    
        image = check_image_shape(image, logger) 
        
        
        if settings.analysis_method == "Dendrite Specific":
            spine_summary = spine_and_dendrite_processing(image, labels, spine_summary, settings, locations, files[file], log, logger)
        else:
            spine_summary = spine_and_whole_neuron_processing(image, labels, spine_summary, settings, locations, files[file], log, logger)
        
        
        if settings.shape_error == True:
            logger.info(f"!!! One or more images moved to \\Not_Processed due to having\nless than 5 Z slices. Please modify these files before reprocessing.\n")
        
        spine_summary.to_csv(locations.tables + 'Detected_spines_summary.csv',index=False) 
    
    logger.info("SpinePipe analysis complete.")
          


def spine_and_whole_neuron_processing(image, labels, spine_summary, settings, locations, filename, log, logger):
    neuron = image[:,settings.neuron_channel-1,:,:]

    
    spines = (labels == 1)
    dendrites = (labels == 2)
    soma = (labels==3)
    
    if np.max(soma) == 0:
        soma_distance = soma
    else:
        soma_distance = ndimage.distance_transform_edt(np.invert(soma))
    

    #fitler out small dendites
    dendrite_labels, num_detected = ndimage.label(dendrites)
    
    
    # Calculate volumes and filter
    dend_vols = ndimage.sum_labels(dendrites, dendrite_labels, index=range(1, num_detected + 1))

    large_dendrites = dend_vols >= settings.min_dendrite_vol

    # Create new dendrite binary
    dendrites = np.isin(dendrite_labels, np.nonzero(large_dendrites)[0] + 1).astype(bool)

    filt_dendrites = np.max(measure.label(dendrites))
    
    logger.info(f"  {filt_dendrites} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")
           
    
    
    if filt_dendrites > 0:
    
        logger.info(f"   Processing {filt_dendrites} dendrites...")    
        #Create Distance Map

        dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites)) #invert neuron mask to get outside distance  
        dendrites = dendrites.astype(np.uint8)
        skeleton = morphology.skeletonize_3d(dendrites)
        
        
        #if settings.save_val_data == True:
        #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)
        
        #Create Neuron MIP for validation - include distance map too
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
    
        #Detection
        logger.info("   Detecting spines...")
        spine_labels = spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
        
        #logger.info(f" {np.max(spine_labels)}.")
        #max_label = np.max(spine_labels)
    
        #Measurements
        #Create 4D Labels
        tifffile.imwrite(locations.tables + 'Detected_spines.tif', spine_labels.astype(np.uint16), imagej=True, photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
        
        spine_table, spines_filtered = spine_measurementsV2(image, spine_labels, 1, 0, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)
        #spine_table, spines_filtered = spine_measurementsV1(image, spine_labels, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)
        
        #Create 4D Labels
        tifffile.imwrite(locations.tables + 'Detected_spines_filtered.tif', spines_filtered.astype(np.uint16), imagej=True, photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})                                                  #soma_mask, soma_distance, )
        
        dendrite_length = np.sum(skeleton == 1)
        dendrite_volume = np.sum(dendrites ==1)
        
          
        if len(spine_table) == 0:
            logger.info(f"  *No spines were analyzed for this image.")
            
        else:
            
            
             
            neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, spines_filtered, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+filename, 'float', settings)
            
            logger.info("   Creating spine arrays on GPU...")
            #Extract MIPs for each spine
            spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image, labels, spines_filtered, spine_table, settings.roi_volume_size, settings, locations, filename,  logger, settings.GPU_block_size)
            
                
            #use the spine_MIPs to measure spine area        
            label_areas = spine_MIPs[:, 1, :, :]
            spine_areas = np.sum(label_areas > 0, axis=(1, 2))
            df_spine_areas = pd.DataFrame({'spine_area': spine_areas})
            
            df_spine_areas['label'] = spine_table['label'].values
            # Reindex df_spine_areas to match the index of spine_table
            #df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
            #df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False) 
            spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
            spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts 
            #spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])
            spine_table.insert(6, 'spine_area_um2', spine_table['spine_area'] * (settings.input_resXY **2))
            
        
        
            spine_table.drop(['dendrite_id'], axis=1, inplace=True)
    
            spine_table.to_csv(locations.tables + 'Detected_spines_'+filename+'.csv',index=False) 
        
            #create summary
            summary = create_spine_summary_neuron(spine_table, filename, dendrite_length, dendrite_volume, settings)
    
            # Append to the overall summary DataFrame
            spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
    else:
        logger.info("  *No dendrites were analyzed for this image.")
    
    logger.info(f" Processing complete for file {filename}\n---")
    
    return spine_summary
    
def spine_and_dendrite_processing(image, labels, spine_summary, settings, locations, filename, log, logger):       
    neuron = image[:,settings.neuron_channel-1,:,:]

    
    spines = (labels == 1)
    all_dendrites = (labels == 2)
    #soma = (image == 3).astype(np.uint8)
    spines_orig = spines * 65535
    
    soma = (labels==3)
    if np.max(soma) == 5:
        soma_distance = soma
    else:
        soma_distance = ndimage.distance_transform_edt(np.invert(soma))
    
    all_filtered_spines = np.zeros_like(spines)
    all_skeletons = np.zeros_like(spines)
    
    all_filtered_spines_table = pd.DataFrame()
    
    
    #dendrite_labels = measure.label(all_dendrites)
    #fitler out small dendites
    dendrite_labels, num_detected = ndimage.label(all_dendrites)
       
    
    # Calculate volumes and filter
    dend_vols = ndimage.sum_labels(all_dendrites, dendrite_labels, index=range(1, num_detected + 1))

    large_dendrites = dend_vols >= settings.min_dendrite_vol

    # Create new dendrite binary
    filtered_dendrites = np.isin(dendrite_labels, np.nonzero(large_dendrites)[0] + 1).astype(bool)
    
    dendrite_labels = measure.label(filtered_dendrites)
    
    logger.info(f"  {np.max(dendrite_labels)} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")
 
       
    logger.info(f"  Processing {np.max(dendrite_labels)} dendrites...")
    
    
    
    for dendrite in range(1, np.max(dendrite_labels)+1):
        logger.info(f"   Detecting spines on dendrite {dendrite}...")
        dendrite_mask = dendrite_labels == dendrite
        
        if np.sum(dendrite_mask) > settings.min_dendrite_vol:
            
        
            #Create Distance Map

            dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrite_mask)) #invert neuron mask to get outside distance  
            dendrite_mask = dendrite_mask.astype(np.uint8)
            skeleton = morphology.skeletonize_3d(dendrite_mask)
            all_skeletons = skeleton + all_skeletons
            #if settings.save_val_data == True:
            #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)
            
            #Create Neuron MIP for validation - include distance map too
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
            
           
            
            #Detection
            
            spine_labels = spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
            
            #tifffile.imwrite(locations.MIPs+"spines_"+str(dendrite)+files[file], spine_labels.astype(np.uint16), imagej=True, photometric='minisblack',
            #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
                    
            #offset object counts to ensure unique ids 
            max_label = np.max(all_filtered_spines)
        
            #logger.info(max_label)
            #Measurements
            spine_table, spines_filtered = spine_measurementsV2(image, spine_labels, dendrite, max_label, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)
                                                              #soma_mask, soma_distance, )
            
            dendrite_length = np.sum(skeleton == 1)
            spine_table.insert(5, 'dendrite_length', dendrite_length)
            dendrite_volume = np.sum(dendrite_mask ==1)
            spine_table.insert(6, 'dendrite_vol', dendrite_volume)
            #tifffile.imwrite(locations.MIPs+"filtered_"+str(dendrite)+files[file], spines_filtered.astype(np.uint16), imagej=True, photometric='minisblack',
            #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
            
            
            #offset detected objects in image and add to all spines
            
            #logger.info(np.max(spines_filtered))
            all_filtered_spines = all_filtered_spines + spines_filtered
            #logger.info(f"{all_filtered_spines.shape}  {spines_filtered.shape}")

            #remove detected spines from original spine image to prevent double counting
            spines[spines_filtered > 0] = 0
                        
            all_filtered_spines_table = pd.concat([all_filtered_spines_table, spine_table], ignore_index=True)


    
    #spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
    
    
    #relable the label image
    # Get unique labels while preserving the order
    #unique_labels = np.unique(all_filtered_spines[all_filtered_spines > 0])
    #if unique_labels 
    #logger.info(unique_labels)
    # Create a mapping from old labels to new sequential labels
    #label_mapping = {labelx: i for i, labelx in enumerate(unique_labels, start=1)}

    # Apply this mapping to the image to create a relabelled image
    #relabelled_spines = np.copy(all_filtered_spines)
    #for old_label, new_label in label_mapping.items():
    #    relabelled_spines[relabelled_spines == old_label] = new_label
    
    if 'dendrite_distance' in locals():
        neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines_orig, all_filtered_spines, dendrite_labels, all_skeletons, dendrite_distance], locations.MIPs+"MIP_"+filename, 'float', settings)
    else:
        logger.info("  *No dendrites were analyzed for this image.")
    
    if len(all_filtered_spines_table) == 0:
        logger.info("  *No spines were analyzed for this image.")
        
    else:
        #all_filtered_spines_table['label'] = all_filtered_spines_table['label'].map(label_mapping)
   
        logger.info("  Creating spine arrays on GPU...")
        #Extract MIPs for each spine
        spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image, labels, all_filtered_spines, all_filtered_spines_table, settings.roi_volume_size, settings, locations, filename,  logger, settings.GPU_block_size)

        all_filtered_spines_table.to_csv(locations.tables + 'Detected_spines_'+filename+'pre.csv',index=False)         

        #perform 2D spine refinement here Unet detecting spine neck and head

        #use the spine_MIPs to measure spine area        
        label_areas = spine_MIPs[:, 1, :, :]
        spine_areas = np.sum(label_areas > 0, axis=(1, 2))
        df_spine_areas = pd.DataFrame({'spine_area': spine_areas})
        
        df_spine_areas['label'] = all_filtered_spines_table['label'].values
        
        df_spine_areas.to_csv(locations.tables + 'Detected_spines_'+filename+'areas.csv',index=False) 
        
        all_filtered_spines_table = all_filtered_spines_table.merge(df_spine_areas, on='label', how='left')
        all_filtered_spines_table.insert(7, 'spine_area', all_filtered_spines_table.pop('spine_area')) #pops and inserts 
        #all_filtered_spines_table.insert(7, 'spine_area', df_spine_areas['spine_area'])
        all_filtered_spines_table.insert(8, 'spine_area_um2', all_filtered_spines_table['spine_area'] * (settings.input_resXY **2))
        
        ##add the averages to the summary        
        #if 'avg_spine_area' not in spine_summary.columns:
        #    spine_summary.insert(12, 'avg_spine_area', all_filtered_spines_table['spine_area'].mean())
        #else:
        #    spine_summary.at[spine_summary.index[-1], 'avg_spine_area'] = all_filtered_spines_table['spine_area'].mean()
         
        #if 'avg_spine_area_um2' not in spine_summary.columns:
        #    spine_summary.insert(13, 'avg_spine_area_um2', all_filtered_spines_table['spine_area_um2'].mean())
        #else:
        #    spine_summary.at[spine_summary.index[-1], 'avg_spine_area_um2'] = all_filtered_spines_table['spine_area_um2'].mean()
            

        
        #create summary
        summary = create_spine_summary_dendrite(all_filtered_spines_table, filename, settings)
        
        all_filtered_spines_table.drop(['dendrite_length', 'dendrite_vol'], axis=1, inplace=True)

        all_filtered_spines_table.to_csv(locations.tables + 'Detected_spines_'+filename+'.csv',index=False) 
        
        
        
        

    spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
    
    logger.info(f"  \nProcessing complete for file {filename}\n---")

    return spine_summary


def create_spine_summary_neuron(filtered_table, filename, dendrite_length, dendrite_volume, settings):
    #create summary table
   
    #spine_reduced = filtered_table.drop(columns=['label', 'z', 'y', 'x'])
    updated_table = filtered_table.iloc[:, 4:]
    

    #spine_summary = updated_table.mean()
    spine_summary = pd.DataFrame([updated_table.mean()])
    #spine_summary = summary.groupby('dendrite_id').mean()
    #spine_counts = summary.groupby('dendrite_id').size()
    
    spine_summary = spine_summary.add_prefix('avg_')
    #spine_summary.reset_index(inplace=True)
    #spine_summary.index = spine_summary.index + 1
    # update summary with additional metrics
    spine_summary.insert(0, 'Filename', filename)  # Insert a column at the beginning
    spine_summary.insert(1, 'res_XY', settings.input_resXY)  
    spine_summary.insert(2, 'res_Z', settings.input_resZ)
    spine_summary.insert(3, 'dendrite_length', dendrite_length)
    spine_summary.insert(4, 'dendrite_length_um', dendrite_length * settings.input_resXY)
    spine_summary.insert(5, 'dendrite_vol', dendrite_volume)
    spine_summary.insert(6, 'dendrite_vol_um3', dendrite_volume * settings.input_resXY*settings.input_resXY*settings.input_resZ)
    spine_summary.insert(7, 'total_spines', filtered_table.shape[0])
    spine_summary.insert(8, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length_um'])
    spine_summary.insert(9, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_vol_um3'])
    
    return spine_summary
    

def create_spine_summary_dendrite(filtered_table, filename, settings):
    #create summary table
   
    #spine_reduced = filtered_table.drop(columns=['label', 'z', 'y', 'x'])
    summary = filtered_table.iloc[:, 4:]

    
    spine_summary = summary.groupby('dendrite_id').mean()
    spine_counts = summary.groupby('dendrite_id').size()
    
    spine_summary = spine_summary.add_prefix('avg_')
    spine_summary.reset_index(inplace=True)
    spine_summary.index = spine_summary.index + 1
    # update summary with additional metrics
    spine_summary.insert(0, 'Filename', filename)  # Insert a column at the beginning
    spine_summary.insert(1, 'res_XY', settings.input_resXY)  
    spine_summary.insert(2, 'res_Z', settings.input_resZ)
    spine_summary.insert(6, 'total_spines', spine_counts)
    spine_summary.rename(columns={'avg_dendrite_length': 'dendrite_length'}, inplace=True)
    spine_summary.rename(columns={'avg_dendrite_vol': 'dendrite_vol'}, inplace=True)
    spine_summary['dendrite_length_um'] = spine_summary['dendrite_length'] * settings.input_resXY
    spine_summary['dendrite_vol_um3'] = spine_summary['dendrite_vol'] *settings.input_resXY*settings.input_resXY*settings.input_resZ
    spine_summary = move_column(spine_summary, 'dendrite_length_um', 5)
    spine_summary = move_column(spine_summary, 'dendrite_vol_um3', 7)
    spine_summary.insert(9, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length_um'])
    spine_summary.insert(10, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_vol_um3'])
    
    
    return spine_summary
            

 
def analyze_spines_4D(settings, locations, log, logger):
    logger.info("Analyzing spines across time...")
    #spines = 1
    #dendrites = 2
    #soma = 3
    
    datasetname = os.path.basename(os.path.normpath(locations.input_dir))
    
    image = imread(locations.input_dir+"/Registered/Registered_images_4D.tif")
    
    labels = imread(locations.input_dir+"/Registered/Registered_labels_4D.tif")
    
    if image.shape != labels.shape:
        logger.info(log)
        raise RuntimeError("Image and labels are not the same shape.")
    
    #Currently registering one channel - but need to add capacity to deal with multi channels
    #in that situation the data will be CMZYX
    #if only one channel then MZYX, so if if shape only 4 add a empty axis at the beginning and then proceed
    #Add channel axis if not present
    if len(image.shape) == 4:
        image = np.expand_dims(image, axis=2)
    
    #select neuron channel
    neuron = image[:,:,settings.neuron_channel-1,:,:]
    
    
    spines = (labels == 1)
    dendrites = (labels == 2)
    soma = (labels==3)
    
    #
    spine_labels = spine_detection_4d(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #value used to remove small holes
    
    
    #logger.info(f" {np.max(spine_labels[0,:,:,:])}.")
    
    #create lists for each 3D volume
    dendrite_distance_list = []
    skeleton_list = []
    soma_distance_list = []
        
    spine_summary = pd.DataFrame()
    
    
    
    for t in range(labels.shape[0]):
        
        logger.info(f" Processing timepoint {t+1} of {labels.shape[0]}.")
        
        dendrites_3d = dendrites[t, :, :, :]
        
        #fitler out small dendites
        dendrite_labels, num_detected = ndimage.label(dendrites_3d)
        
        
        # Calculate volumes and filter
        dend_vols = ndimage.sum_labels(dendrites_3d, dendrite_labels, index=range(1, num_detected + 1))

        large_dendrites = dend_vols >= settings.min_dendrite_vol

        # Create new dendrite binary
        dendrites_3d = np.isin(dendrite_labels, np.nonzero(large_dendrites)[0] + 1).astype(bool)

        filt_dendrites = np.max(measure.label(dendrites_3d))
        
        logger.info(f"  {filt_dendrites} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")
               
        if filt_dendrites > 0:
            
            logger.info(f"   Processing {filt_dendrites} dendrites...")
        
        
            #Create Distance Map
            
            
        
            dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites_3d)) #invert neuron mask to get outside distance  
            dendrite_distance_list.append(dendrite_distance)
            
            dendrites_3d = dendrites_3d.astype(np.uint8)
            
            skeleton = morphology.skeletonize_3d(dendrites_3d)
            skeleton_list.append(skeleton)
            
            soma_3d = soma[t, :, :, :]
            
            if np.max(soma_3d) == 0:
                soma_distance = soma[t, :, :, :]
            
            else:
                soma_distance = ndimage.distance_transform_edt(np.invert(soma))
            
           
            #if settings.save_val_data == True:
            #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)
            
            #Create Neuron MIP for validation - include distance map too
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
        
            #Detection
            #logger.info(" Detecting spines...")
        else:
            logger.info("  *No dendrites were analyzed for this image.")
            
        dendrite_distance = np.stack(dendrite_distance_list, axis=0)
        skeleton = np.stack(skeleton_list, axis=0)
        soma_distance = np.stack(dendrite_distance_list, axis=0)
        
        spines_filtered_list = []
        all_spines_table = pd.DataFrame()
        all_summary_table = pd.DataFrame()
    
    #Measurements
    
    
    for t in range(labels.shape[0]):
        logger.info(f" Measuring timepoint {t+1} of {labels.shape[0]}.")
        #max label for 4d?
        spine_table, spines_filtered = spine_measurementsV2(image[t, :, :, :], spine_labels[t, :, :, :], 0, 0, settings.neuron_channel, dendrite_distance[t, :, :, :], soma_distance[t, :, :, :], settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, datasetname, logger)
                                                              #soma_mask, soma_distance, )
        if t == 0:
            previous_spines = set(np.unique(spines_filtered))
            new = 0
            pruned = 0
        
        else:
            current_spines = set(np.unique(spines_filtered))
            new = len(current_spines - previous_spines)
            pruned = len(previous_spines - current_spines)
        
        dendrite_length = np.sum(skeleton[t,:,:,:] == 1)

        dendrite_volume = np.sum(dendrites[t,:,:,:] ==1)
        
        
        if len(spine_table) == 0:
            logger.info(f"  *No spines were analyzed for this image.")
            
        else:
            #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, spines_filtered, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+filename, 'float', settings)
        
            spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image[t, :, :, :], labels[t,:,:,:], spines_filtered, spine_table, settings.roi_volume_size, settings, locations, str(t+1)+'.tif',  logger, settings.GPU_block_size)
            
            spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_1.csv',index=False) 
            
            label_areas = spine_MIPs[:, 1, :, :]
            spine_areas = np.sum(label_areas > 0, axis=(1, 2))
            df_spine_areas = pd.DataFrame({'spine_area': spine_areas})
            
            df_spine_areas['label'] = spine_table['label'].values
            
            df_spine_areas.to_csv(locations.tables + str(t)+ 'Detected_spines_areas.csv',index=False) 
            # Reindex df_spine_areas to match the index of spine_table
            #df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
            #df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False) 
            spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
            spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts 
            #spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])
            spine_table.insert(6, 'spine_area_um2', spine_table['spine_area'] * (settings.input_resXY **2))
            
            spine_table.insert(0, 'timepoint', t+1)
            
            spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_2.csv',index=False) 
            
            #append tables
            all_spines_table = pd.concat([all_spines_table, spine_table], ignore_index=True)
            
            all_spines_table.to_csv(locations.tables + str(t)+ 'Detected_spines_appened.csv',index=False) 
            
            summary = create_spine_summary_neuron(spine_table, str(t+1), dendrite_length, dendrite_volume, settings)
            
            summary.insert(7, 'new_spines', new)
            summary.insert(8, 'pruned_spines', pruned)
            
            # Append to the overall summary DataFrame
            all_summary_table = pd.concat([all_summary_table, summary], ignore_index=True)
            #append images
            spines_filtered_list.append(spines_filtered)
            spines_filtered_all = np.stack(spines_filtered_list, axis=0)
    
     
    #Create Pivot Tables for volume and coordinates
    all_spines_table.drop(['dendrite_id'], axis=1, inplace=True)
    all_spines_table.rename(columns={'label': 'spine_id'}, inplace=True)
    all_summary_table.drop(['avg_dendrite_id'], axis=1, inplace=True)
    all_summary_table.rename(columns={'Filename': 'timepoint'}, inplace=True)
    
    
    vol_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='spine_vol_um3')
    z_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='z')
    y_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='y')
    x_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='x')
    
    #save required tables
    vol_over_t.to_csv(locations.tables + 'Volume_4D.csv',index=False) 
    all_summary_table.to_csv(locations.tables + 'Detected_spines_4D_summary.csv',index=False) 
    all_spines_table.to_csv(locations.tables + 'Detected_spines_4D.csv',index=False) 
    
    
    # Create MIP
    neuron_MIP = create_mip_and_save_multichannel_tiff_4d([neuron, spines, spines_filtered_all, dendrites, skeleton, dendrite_distance], locations.input_dir+"/Registered/Registered_MIPs_4D.tif", 'float', settings)
    
    #Create 4D Labels
    tifffile.imwrite(locations.tables + 'Detected_spines.tif', spines_filtered_all.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'TZYX'})
    
    #Extract MIPs for each spine
    #if len(y_over_t) >=1:
    #    spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

    #Cleanup
    #if os.path.exists(locations.nnUnet_input): shutil.rmtree(locations.nnUnet_input)
    if os.path.exists(locations.MIPs): shutil.rmtree(locations.MIPs)
    if os.path.exists(locations.validation_dir+"/Registered_segmentation_labels/"): shutil.rmtree(locations.validation_dir+"/Registered_segmentation_labels/")
    
    
    #logger.info("Spine analysis complete.\n")


def create_mip_and_save_multichannel_tiff(images_3d, filename, bitdepth, settings):
    """
    Create MIPs from a list of 3D images, merge them into a multi-channel 2D image,
    save as a 16-bit TIFF file, and return the merged 2D multi-channel image.

    Args:
        images_3d (list of numpy.ndarray): List of 3D numpy arrays representing input images.
        filename (str): Filename for the output TIFF file.

    Returns:
        numpy.ndarray: Merged 2D multi-channel image as a numpy array.
    """
    # Create MIPs from the 3D images
    mips = [np.amax(img, axis=0) for img in images_3d]
    
    # Convert the MIPs to a single multichannel image
    multichannel_image = np.stack(mips, axis=0)
    
    # Convert the multichannel image to 16-bit
    multichannel_image = multichannel_image.astype(np.uint16)
    #multichannel_image = rescale_all_channels_to_full_range(multichannel_image)
    
    # Save the multichannel image as a 16-bit TIFF file
    #tifffile.imwrite(filename, multichannel_image, photometric='minisblack')
    
    tifffile.imwrite(filename, multichannel_image, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'CYX'})

    # Return the merged 2D multi-channel image as a numpy array
    return multichannel_image

def create_mip_and_save_multichannel_tiff_4d(images_3d, filename, bitdepth, settings):
    """
    Create MIPs from a list of 3D images, merge them into a multi-channel 2D image,
    save as a 16-bit TIFF file, and return the merged 2D multi-channel image.

    Args:
        images_3d (list of numpy.ndarray): List of 3D numpy arrays representing input images.
        filename (str): Filename for the output TIFF file.

    Returns:
        numpy.ndarray: Merged 2D multi-channel image as a numpy array.
    """
    # Create MIPs from the 3D images
    mips = [np.amax(img, axis=1) for img in images_3d]
    
    # Convert the MIPs to a single multichannel image
    multichannel_image = np.stack(mips, axis=0)
    multichannel_image = np.swapaxes(multichannel_image, 0, 1)
    
    # Convert the multichannel image to 16-bit
    multichannel_image = multichannel_image.astype(np.uint16)
    #multichannel_image = rescale_all_channels_to_full_range(multichannel_image)
    
    # Save the multichannel image as a 16-bit TIFF file
    #tifffile.imwrite(filename, multichannel_image, photometric='minisblack')
    
    tifffile.imwrite(filename, multichannel_image, imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'TCYX'})

    # Return the merged 2D multi-channel image as a numpy array
    return multichannel_image

def rescale_all_channels_to_full_range(array):
    """
    Rescale all channels in a multi-channel numpy array to use the full range of 16-bit values.

    Args:
        array (numpy.ndarray): Input multi-channel numpy array.

    Returns:
        numpy.ndarray: Rescaled multi-channel numpy array.
    """
    num_channels = array.shape[0]

    for channel in range(num_channels):
        # Calculate the minimum and maximum values of the current channel
        min_val = np.min(array[channel])
        max_val = np.max(array[channel])

        # Rescale the current channel to the 16-bit range [0, 65535] using a linear transformation
        array[channel] = (array[channel] - min_val) / (max_val - min_val) * 65535

        # Convert the rescaled channel to uint16 data type
        array[channel] = array[channel].astype(np.uint16)

    return array

# Detects immune cells for multiclass unet output
def spine_detection(spines, erode, remove_borders, logger):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    spines_clean = morphology.remove_small_holes(spines, holes)
    
    if erode[0] > 0:
        #Erode
        #element = morphology.ball(erode)
        ellipsoidal_element = create_ellipsoidal_element(erode[0], erode[1], erode[2])

        spines_eroded = ndimage.binary_erosion(spines, ellipsoidal_element)
        
        # Distance Transform to mark centers
        distance = ndimage.distance_transform_edt(spines_eroded)
        seeds = ndimage.label(distance > 0.1 * distance.max())[0]
    
        #Watershed
        labels = segmentation.watershed(-distance, seeds, mask=spines)
        
    else:
        labels = measure.label(spines)
        
    #Remove objects touching border
    if remove_borders == True:
        padded = np.pad(
          labels,
          ((1, 1), (0, 0), (0, 0)),
          mode='constant',
          constant_values=0,
          )
        labels = segmentation.clear_border(padded)[1:-1]
        
    # add new axis for labels
    #labels = labels[:, np.newaxis, :, :]

    return labels


def spine_detection_4d(spines, erode, remove_borders, logger):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    spines_clean = morphology.remove_small_holes(spines, holes)
    if erode[0] > 0:
        
        #Erode
        #element = morphology.ball(erode)
        ellipsoidal_element = create_ellipsoidal_element(erode[0], erode[1], erode[2])
        ellipsoidal_element = ellipsoidal_element[np.newaxis, :, :, :]
        
        spines_eroded = ndimage.binary_erosion(spines, ellipsoidal_element)
    
        # Distance Transform to mark centers
        distance = ndimage.distance_transform_edt(spines_eroded)
        seeds = ndimage.label(distance > 0.1 * distance.max())[0]

        #Watershed
        labels = segmentation.watershed(-distance, seeds, mask=spines)
        
    else:
        labels = measure.label(spines)
       
    
    if remove_borders == True:
        spines_list = []
        #Remove objects touching border
        for t in range(labels.shape[0]):
            labels_3d = labels[t, :, :, :]
            padded = np.pad(
              labels_3d,
              ((1, 1), (0, 0), (0, 0)),
              mode='constant',
              constant_values=0,
              )
            labels_3d = segmentation.clear_border(padded)[1:-1]
            spines_list.append(labels_3d)
            
        labels = np.stack(spines_list, axis=0)
    # Check if the sequence of labels is continuous
    unique_labels = np.unique(labels)
    if np.all(np.diff(unique_labels) == 1) != True:
        labels = measure.label(labels > 0, background=0)   

    return labels

def create_ellipsoidal_element(radius_z, radius_y, radius_x):
    # Create a grid of points
    z = np.arange(-radius_x, radius_x + 1)
    y = np.arange(-radius_y, radius_y + 1)
    x = np.arange(-radius_z, radius_z + 1)
    z, y, x = np.meshgrid(z, y, x, indexing='ij')

    # Create an ellipsoid
    ellipsoid = (z**2 / radius_z**2) + (y**2 / radius_y**2) + (x**2 / radius_x**2) <= 1

    return ellipsoid


def move_column(df, column, position):
    """
    Move a column in a DataFrame to a specified position.
    
    Parameters:
    - df: pandas.DataFrame.
    - column: The name of the column to move.
    - position: The new position (index) for the column (0-based).
    
    Returns:
    - DataFrame with the column moved to the new position.
    """
    cols = list(df.columns)
    cols.insert(position, cols.pop(cols.index(column)))
    return df[cols]


def spine_measurementsV2(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, soma_distance, sizes, dist, settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)
        
    Returns:
        pandas table and labeled spine image
    """


    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    #Measure channel 1:
    logger.info("    Measuring channel 1...")
    #logger.info(f" {labels.shape}, {image.shape}")
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image[:,0,:,:],
            properties=['label', 'centroid', 'area', 'mean_intensity', 'max_intensity'], #area is volume for 3D images
            )
        )
    
    #rename mean intensity
    main_table.rename(columns={'mean_intensity':'C1_mean_int'}, inplace=True)
    main_table.rename(columns={'max_intensity':'C1_max_int'}, inplace=True)
    main_table.rename(columns={'centroid-0':'z'}, inplace=True)
    main_table.rename(columns={'centroid-1':'y'}, inplace=True)
    main_table.rename(columns={'centroid-2':'x'}, inplace=True)
    
    
    
    # measure remaining channels
    for ch in range(image.shape[1]-1):
        logger.info(f"    Measuring channel {ch+2}...")
        #Measure
        table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:,ch+1,:,:],
                properties=['label', 'mean_intensity', 'max_intensity'], #area is volume for 3D images
            )
        )
        
        #rename mean intensity
        table.rename(columns={'mean_intensity':'C'+str(ch+2)+'_mean_int'}, inplace=True)
        table.rename(columns={'max_intensity':'C'+str(ch+2)+'_max_int'}, inplace=True)
        Mean = table['C'+str(ch+2)+'_mean_int']
        Max = table['C'+str(ch+2)+'_max_int']
        
        #combine columns with main table
        main_table = main_table.join(Mean)
        main_table = main_table.join(Max)


    #measure distance to dendrite
    logger.info("    Measuring distances to dendrite...")
    #logger.info(f" {labels.shape}, {dendrite_distance.shape}")
    distance_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=dendrite_distance,
            properties=['label', 'min_intensity', 'max_intensity'], #area is volume for 3D images
        )
    )

    #rename distance column
    distance_table.rename(columns={'min_intensity':'dist_to_dendrite'}, inplace=True)
    distance_table.rename(columns={'max_intensity':'spine_length'}, inplace=True)
    
    distance_col = distance_table["dist_to_dendrite"]
    main_table = main_table.join(distance_col)
    distance_col = distance_table["spine_length"]
    main_table = main_table.join(distance_col)
    
    if np.max(soma_distance) > 0:
    #measure distance to dendrite
        logger.info("    Measuring distances to soma...")
        distance_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=soma_distance,
                properties=['label', 'min_intensity', 'max_intensity'], #area is volume for 3D images
            )
        )
        distance_table.rename(columns={'min_intensity':'dist_to_soma'}, inplace=True)
        distance_col = distance_table["dist_to_soma"]
        main_table = main_table.join(distance_col)
    else:
        main_table['dist_to_soma'] = pd.NA
    
    
    #filter out small objects
    volume_min = sizes[0] #3
    volume_max = sizes[1] #1500?
    
    #logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")
    

    #filter based on volume
    #logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)
    
   
    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max) ] 
    

    logger.info(f"    Spines before volume filter = {spinebefore}. After volume filter = {len(filtered_table)}. ")
    #logger.info(f"  filtered table after area = {len(filtered_table)}")
    
    #filter based on distance to dendrite
    spinebefore = len(filtered_table)
    #logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    #logger.info(f"  filtered table before dist = {len(filtered_table)}. and distance = {dist}")
    filtered_table = filtered_table[(filtered_table['spine_length'] < dist)] 
    logger.info(f"    Spines before distance filter = {spinebefore}. After distance filter = {len(filtered_table)}. ")
    
    if settings.Track != True:
        #update label numbers based on offset
        filtered_table['label'] += max_label
        labels[labels > 0] += max_label
        labels = create_filtered_labels_image(labels, filtered_table, logger)
    else:
        
        #Clean up label image to remove objects from image.
        ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask 
        mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        labels = np.where(mask_to_keep, labels, 0)
    
    
    
    
    
    

    
    #update to included dendrite_id
    filtered_table.insert(4, 'dendrite_id', dendrite)
    
    
    #create vol um measurement
    filtered_table.insert(6, 'spine_vol_um3', filtered_table['area'] * (settings.input_resXY*settings.input_resXY*settings.input_resZ))
    filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)
    
    #create dist um cols

    filtered_table = move_column(filtered_table, 'spine_length', 7)
    filtered_table.insert(8, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY))
    filtered_table = move_column(filtered_table, 'dist_to_dendrite', 9)
    filtered_table.insert(10, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY))
    filtered_table = move_column(filtered_table, 'dist_to_soma', 11)
    filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))
    
    #logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    #logger.info(f"  image labels before filter = {np.max(labels)}.")
    
   
    
    logger.info(f"    After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)}")
   
  
    return filtered_table, labels



def create_filtered_labels_image(labels, filtered_table, logger):
    """
    Create a new labels image using a filtered regionprops table, more efficiently.

    Args:
        labels (numpy.ndarray): Input labels image.
        filtered_table (pd.DataFrame): Filtered regionprops table.

    Returns:
        numpy.ndarray: Filtered labels image.
    """

    filtered_labels_list = filtered_table['label'].astype(labels.dtype).values
    
    #logger.info(f" {np.max(labels)} , {len(filtered_labels_list)}")
    #logger.info(filtered_labels_list)
    # Create a mask that retains only the labels present in the filtered_table
    mask = np.isin(labels, filtered_labels_list)
    # Create the filtered labels image by multiplying the mask with the original labels
    filtered_labels = np.where(mask, labels, 0)    
    #logger.info(np.max(filtered_labels))
    return filtered_labels

def create_filtered_and_unfiltered_spine_arrays_cupy(image, spines_filtered, labels, table, volume_size, settings, locations, file, logger):
    merge_mip_list = [] #image MIP
    merge_slice_list = [] # image slize
    merge_vol_list = []
    #mip_label_filtered_list = [] #Mip filtered by label - we don't need this
    #slice_before_mip_list = [] #slice filtered by label - we don't need this either
    
    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)
    
    image_cp = cp.array(image)
    spines_filtered_cp = cp.array(spines_filtered)
    labels_cp = cp.array(labels)
    
    volume_size_z = int(volume_size / settings.input_resZ)
    volume_size_y = int(volume_size / settings.input_resXY)
    volume_size_x = int(volume_size / settings.input_resXY)


    # Pad the image and spines_filtered and keep the channel axis intact for the image
    image_cp = cp.pad(image_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2), (0, 0)), mode='constant', constant_values=0)
    spines_filtered_cp = cp.pad(spines_filtered_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2)), mode='constant', constant_values=0)
    labels_cp = cp.pad(labels_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2)), mode='constant', constant_values=0)


    for index, row in table.iterrows():
        #logger.info(f' Extracting and saving 3D ROIs for cell {index + 1}/{len(table)}', end='\r')
        z, y, x, label = int(row['z']), int(row['y']), int(row['x']), int(row['label'])
        
        z_min, z_max = z - volume_size_z // 2, z + volume_size_z // 2
        y_min, y_max = y - volume_size_y // 2, y + volume_size_y // 2
        x_min, x_max = x - volume_size_x // 2, x + volume_size_x // 2
        
        # Extract the volume and keep the channel axis intact
        image_vol = image_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2, :]
        spine_vol = spines_filtered_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2]
        label_vol = labels_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2]

        # Filter the volume to only show image data inside the label
        spine_mask = (spine_vol == label)
        spine_vol = spine_vol * spine_mask
        
        spine_vol = cp.expand_dims(spine_vol, axis=-1)
        label_vol = cp.expand_dims(label_vol, axis=-1)
        spine_mask = cp.expand_dims(spine_mask, axis=-1)
        
        #logger.info(f' image{image_vol.shape} ,spine_vol.shape {spine_vol.shape}labelvol shape {label_vol.shape}, spine maskshape {spine_mask.shape} ')
        
        #for the labels we actually want the dendrite label, but only the spine label for the center spine - requires masking spine label, but not dendrite label
        #cleaned_label = cp.copy(extracted_label)
        #clear outside spine for spine label only - leave dendrite, soma
        label_vol[(label_vol == 1) & (spine_mask == 0)] = 0

        #
        
        # Extract 2D slice at position z
        image_slice = image_vol[volume_size_z // 2, :, :, :]
        spine_slice = spine_vol[volume_size_z // 2, :, :, :]
        label_slice = label_vol[volume_size_z // 2, :, :, :]

        #taking out the masked arrays - I don't think we need these        
        
        #why expanding -taking htis out temporarily
        ##spine_mask_expanded = cp.expand_dims(spine_mask, axis=-1)
        
        #extracted_volume_label_filtered = extracted_volume * spine_mask_expanded

        # Extract 2D slice before the MIP in step 3 is created
        #slice_before_mip = extracted_volume_label_filtered[volume_size_z // 2, :, :, :]
        #spine_slice = spine_mask_expanded[volume_size_z // 2, :, :, :]

        # Compute MIPs - this could be done by merging then mip but leave as this for now
        image_mip = cp.max(image_vol, axis=0)
        image_mip = image_mip[cp.newaxis, :, :] #expand to add label
        spine_mip=cp.max(spine_vol, axis = 0)
        spine_mip = spine_mip[cp.newaxis, :, :] # *65535#expand to add label
        label_mip=cp.max(label_vol, axis = 0)
        label_mip = label_mip[cp.newaxis, :, :]
        
        #logger.info(f' image mip {image_mip.shape} ,spine mip.shape {spine_mip.shape} label mip shape {label_mip.shape} ')
        
        merge_mip = cp.concatenate((image_mip, spine_mip, label_mip), axis=0)
        
        merge_mip_list.append(merge_mip.get())
        
        image_slice = image_slice[cp.newaxis, :, :]#expand to add label
        spine_slice = spine_slice[cp.newaxis, :, :]
        label_slice = label_slice[cp.newaxis, :, :]
        
        #logger.info(f' image slice {image_slice.shape} ,spine slice.shape {spine_slice.shape} label slice shape {label_slice.shape} ')
        
        merge_slice = cp.concatenate((image_slice, spine_slice, label_slice), axis=0)
        
        merge_slice_list.append(merge_slice.get())
        
        # now add Label slice to image slice
        
        #mip_label_filtered = cp.max(extracted_volume_label_filtered, axis=0)
        #mip_label_filtered = mip_label_filtered[cp.newaxis, :, :] #expand to add label
        #spine_slice = spine_slice[cp.newaxis, :, :] *65535 #expand to add label
        #mip_label_filtered= cp.concatenate((mip_label_filtered, spine_slice), axis=0)
        
        #mip_label_filtered_list.append(mip_label_filtered.get())
        
        #slice_before_mip = slice_before_mip[cp.newaxis, :, :] #expand to add label
        #slice_before_mip= cp.concatenate((slice_before_mip, spine_slice), axis=0)
        
        #slice_before_mip_list.append(slice_before_mip.get())
        
        #Could potentially add axis to volumes at beginning, then do all the mips w/o expansion - would be cleaner merge on axis 1 instead of 0
        # ie.. merge vol then create MIP and slice- come back and clean up if time...
        #CZYX
        image_vol = cp.expand_dims(image_vol, axis=0)
        #image_vol = image_vol[cp.newaxis, :, :, :]
        spine_vol = cp.expand_dims(spine_vol, axis=0)
        #spine_vol = image_vol[cp.newaxis, :, :, :, :]
        label_vol = cp.expand_dims(label_vol, axis=0)
        
        #logger.info(f' image{image_vol.shape} ,spine_vol.shape {spine_vol.shape}labelvol shape {label_vol.shape} ')
        
        merge_vol = cp.concatenate((image_vol, spine_vol, label_vol), axis=0)
        merge_vol_list.append(merge_vol.get())
                
        
        del image_vol, label_vol, spine_vol, image_slice, spine_slice, label_slice, image_mip, spine_mip, label_mip, merge_vol
        cp.cuda.Stream.null.synchronize()
        gc.collect()

    mip_array = np.stack(merge_mip_list)
    #mip_array = np.moveaxis(mip_array, 3, 1)
    mip_array = mip_array.squeeze(axis=-1)
    
    slice_array = np.stack(merge_slice_list)
    #slice_z_array = np.moveaxis(slice_z_array, 3, 1)
    slice_array = slice_array.squeeze(axis=-1)
    
    vol_array = np.stack(merge_vol_list)
    vol_array = vol_array.squeeze(axis=-1)
    
    #mip_label_filtered_array = np.stack(mip_label_filtered_list)
    #mip_label_filtered_array = np.moveaxis(mip_label_filtered_array, 3, 1)
    #mip_label_filtered_array = mip_label_filtered_array.squeeze(axis=-1)
    
    #slice_before_mip_array = np.stack(slice_before_mip_list)
    #slice_before_mip_array = np.moveaxis(slice_before_mip_array, 3, 1)
    #slice_before_mip_array = slice_before_mip_array.squeeze(axis=-1)
    
    return mip_array, slice_array, vol_array

def create_spine_arrays_in_blocks(image, labels, spines_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #suppress warning about subtracting from table without copying
    original_chained_assignment = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None
    
    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)
    
    mip_list = []
    slice_list = []
    vol_list = []

    block_size_z, block_size_y, block_size_x = block_size

    z_blocks = math.ceil(image.shape[0] / block_size_z)
    y_blocks = math.ceil(image.shape[1] / block_size_y)
    x_blocks = math.ceil(image.shape[2] / block_size_x)
    total_blocks = z_blocks * y_blocks * x_blocks

    #logger.info(f'    Total blocks used for GPU spine array calculations: {total_blocks} ')

    for i in range(z_blocks):
        for j in range(y_blocks):
            for k in range(x_blocks):
                z_start = i * block_size_z
                z_end = min((i + 1) * block_size_z, image.shape[0])
                y_start = j * block_size_y
                y_end = min((j + 1) * block_size_y, image.shape[1])
                x_start = k * block_size_x
                x_end = min((k + 1) * block_size_x, image.shape[2])

                padding_z = int(max(0, (volume_size // settings.input_resZ) // 2))
                padding_y = int(max(0, (volume_size // settings.input_resXY) // 2))
                padding_x = int(max(0, (volume_size // settings.input_resXY) // 2))

                padded_z_start = int(max(0, z_start - padding_z))
                padded_z_end = int(min(image.shape[0], z_end + padding_z))
                padded_y_start = int(max(0, y_start - padding_y))
                padded_y_end = int(min(image.shape[1], y_end + padding_y))
                padded_x_start = int(max(0, x_start - padding_x))
                padded_x_end = int(min(image.shape[2], x_end + padding_x))
                
                #print(padded_z_start,padded_z_end, padded_y_start, padded_y_end, padded_x_start,padded_x_end)

                block_image = image[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_spines_filtered = spines_filtered[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_labels = labels[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                #print(table['z'])
                block_table = table[(table['z'] >= z_start) & (table['z'] < z_end) & (table['y'] >= y_start) & (table['y'] < y_end) & (table['x'] >= x_start) & (table['x'] < x_end)]
                #print(block_table)
                if len(block_table) > 0:
                    #block_table['z'] = block_table['z'] - padded_z_start
                    #block_table['y'] = block_table['y'] - padded_y_start
                    #block_table['x'] = block_table['x'] - padded_x_start
                    block_table.loc[:, 'z'] = block_table['z'] - padded_z_start
                    block_table.loc[:, 'y'] = block_table['y'] - padded_y_start
                    block_table.loc[:, 'x'] = block_table['x'] - padded_x_start
                    
                    block_mip, block_slice, block_vol = create_filtered_and_unfiltered_spine_arrays_cupy(
                        block_image, block_spines_filtered, block_labels, block_table, volume_size, settings, locations, file, logger
                    )
                    mip_list.extend(block_mip)
                    slice_list.extend(block_slice)
                    vol_list.extend(block_vol)
                    gc.collect()                  
        
        progress_percentage = ((i + 1) * y_blocks * x_blocks) / total_blocks * 100
        #print(f'Progress: {progress_percentage:.2f}%   ', end='', flush=True)

    #print(mip_list)
    # Convert lists to arrays and concatenate
    mip_array = np.stack(mip_list, axis = 0)
    slice_array = np.stack(slice_list, axis=0)
    vol_array = np.stack(vol_list, axis=0)
    
    vol_array = np.transpose(vol_array, (0, 2, 1, 3, 4))
    #mip_label_filtered_array = np.stack(mip_label_filtered_list, axis=0)
    #slice_before_mip_array = np.stack(slice_before_mip_list, axis=0)
    
    #logger.info(f' {mip_array.shape}')
    tifffile.imwrite(locations.arrays+"/Spine_MIPs_"+file, mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    tifffile.imwrite(locations.arrays + "/Spine_slices_" + file, slice_array.astype(np.uint16), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    tifffile.imwrite(locations.arrays + "/Spine_vols_" + file, vol_array.astype(np.uint16), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #tifffile.imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
    #        resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #tifffile.imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #                 metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
    #                 resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    #reenable pandas warning:
    pd.options.mode.chained_assignment = original_chained_assignment
    logger.info(f'    Complete. ')
    return mip_array, slice_array, vol_array

def create_spine_arrays_in_blocks_4d(image, labels_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #suppress warning about subtracting from table without copying
    original_chained_assignment = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None
    
    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)
    
    mip_list = []
    slice_z_list = []
    mip_label_filtered_list = []
    slice_before_mip_list = []

    block_size_z, block_size_y, block_size_x = block_size

    z_blocks = math.ceil(image.shape[0] / block_size_z)
    y_blocks = math.ceil(image.shape[1] / block_size_y)
    x_blocks = math.ceil(image.shape[2] / block_size_x)
    total_blocks = z_blocks * y_blocks * x_blocks

    logger.info(f' Total blocks used for GPU spine array calculations: {total_blocks} ')

    for i in range(z_blocks):
        for j in range(y_blocks):
            for k in range(x_blocks):
                z_start = i * block_size_z
                z_end = min((i + 1) * block_size_z, image.shape[0])
                y_start = j * block_size_y
                y_end = min((j + 1) * block_size_y, image.shape[1])
                x_start = k * block_size_x
                x_end = min((k + 1) * block_size_x, image.shape[2])

                padding_z = int(max(0, (volume_size // settings.input_resZ) // 2))
                padding_y = int(max(0, (volume_size // settings.input_resXY) // 2))
                padding_x = int(max(0, (volume_size // settings.input_resXY) // 2))

                padded_z_start = int(max(0, z_start - padding_z))
                padded_z_end = int(min(image.shape[0], z_end + padding_z))
                padded_y_start = int(max(0, y_start - padding_y))
                padded_y_end = int(min(image.shape[1], y_end + padding_y))
                padded_x_start = int(max(0, x_start - padding_x))
                padded_x_end = int(min(image.shape[2], x_end + padding_x))
                
                #print(padded_z_start,padded_z_end, padded_y_start, padded_y_end, padded_x_start,padded_x_end)

                block_image = image[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                block_labels_filtered = labels_filtered[padded_z_start:padded_z_end, padded_y_start:padded_y_end, padded_x_start:padded_x_end]
                #print(table['z'])
                block_table = table[(table['z'] >= z_start) & (table['z'] < z_end) & (table['y'] >= y_start) & (table['y'] < y_end) & (table['x'] >= x_start) & (table['x'] < x_end)]
                #print(block_table)
                if len(block_table) > 0:
                    #block_table['z'] = block_table['z'] - padded_z_start
                    #block_table['y'] = block_table['y'] - padded_y_start
                    #block_table['x'] = block_table['x'] - padded_x_start
                    block_table.loc[:, 'z'] = block_table['z'] - padded_z_start
                    block_table.loc[:, 'y'] = block_table['y'] - padded_y_start
                    block_table.loc[:, 'x'] = block_table['x'] - padded_x_start
                    
                    block_mip, block_slice_z, block_mip_label_filtered, block_slice_before_mip = create_filtered_and_unfiltered_spine_arrays_cupy(
                        block_image, block_labels_filtered, block_table, volume_size, settings, locations, file, logger
                    )
                    mip_list.extend(block_mip)
                    slice_z_list.extend(block_slice_z)
                    mip_label_filtered_list.extend(block_mip_label_filtered)
                    slice_before_mip_list.extend(block_slice_before_mip)
                    gc.collect()
                    
        
        progress_percentage = ((i + 1) * y_blocks * x_blocks) / total_blocks * 100
        #print(f'Progress: {progress_percentage:.2f}%   ', end='', flush=True)

    #print(mip_list)
    # Convert lists to arrays and concatenate
    mip_array = np.stack(mip_list, axis = 0)
    slice_z_array = np.stack(slice_z_list, axis=0)
    mip_label_filtered_array = np.stack(mip_label_filtered_list, axis=0)
    slice_before_mip_array = np.stack(slice_before_mip_list, axis=0)
    
    tifffile.imwrite(locations.arrays+"/Spines_MIPs_"+file, mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    tifffile.imwrite(locations.arrays + "/Spines_Slices_" + file, slice_z_array.astype(np.uint16), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    tifffile.imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    tifffile.imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    #reenable pandas warning:
    pd.options.mode.chained_assignment = original_chained_assignment
    return mip_array, slice_z_array, mip_label_filtered_array, slice_before_mip_array

def create_MIP_spine_arrays_in_blocks_4d(image, y_locs, x_locs, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

    #image will be TCYX
    
    #we only want raw and object ID (0 and 2)
    
    image = image[:, [0, 2], :, :]
    
    volume_size = int(volume_size // settings.input_resXY)
    # Pad the image
    pad_width = int(volume_size // 2)
    padded_image = np.pad(image, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')

    
    stacked_4D = []
    for (index1, yrow), (index2, xrow) in zip(y_locs.iterrows(), x_locs.iterrows()):
        regions = []
        for t in range(padded_image.shape[0]):       
            if np.isnan(yrow[t]):
                region = np.zeros((padded_image.shape[1], volume_size, volume_size))
            else:
                y_coord = int(yrow[t])
                x_coord = int(xrow[t])
                #mask to object specifically            
                region = padded_image[t, :, y_coord:y_coord+volume_size, x_coord:x_coord+volume_size]

            regions.append(region)
            
    
        # Stack regions horizontally
       
        stacked_regions = np.concatenate(regions, axis=2)
        binary_mask = (stacked_regions[ 1, :, :] == index1)
        binary_mask = binary_mask.astype(int)*100
        stacked_regions[1, :, :] = binary_mask
        stacked_regions = np.expand_dims(stacked_regions, axis=0)
        
        stacked_4D.append(stacked_regions)
    
    # Stack each ID along the Z-axis
    final_4d = np.concatenate(stacked_4D, axis=0)
    
    tifffile.imwrite(locations.input_dir+"/Registered/Isolated_spines_4D.tif", final_4d.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'unit': 'um','axes': 'TCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    return final_4d, final_4d

   
def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))