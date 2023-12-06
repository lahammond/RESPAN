# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""
__title__     = 'SpinePipe'
__version__   = '0.9.4'
__date__      = "19 November, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
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

#import spinepipe.Main.Main as main
#import spinepipe.Main.Timer as timer

#import segmentation_models as sm
#from keras.models import load_model
#from patchify import patchify, unpatchify

import subprocess
#from subprocess import Popen, PIPE

#from tqdm import tqdm

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
from skimage.measure import label
from scipy import ndimage



import gc
#import cupy as cp

#import matplotlib.pyplot as plt
#import logging




##############################################################################
# Main Processing Functions
##############################################################################

def restore_and_segment(inputdir, settings, locations, logger):
    
    if settings.neuron_restore == True:
        restore_neuron(locations.input_dir, settings, locations, logger)
        #include options here for alternative unets if required
        log = nnunet_create_labels(locations.restored, settings, locations, logger)
    else:
        #include options here for alternative unets if required
        log = nnunet_create_labels(locations.input_dir, settings, locations, logger)
        
    return log
    
    
def restore_neuron (inputdir, settings, locations, logger):    
    logger.info("Restoring neuron images...")
        
    files = [file_i for file_i in os.listdir(inputdir) if file_i.endswith('.tif')]
    files = sorted(files)

    for file in range(len(files)):
        logger.info(f' Restoring image {files[file]} \n')

        image = imread(inputdir + file)
        logger.info(f"Raw data has shape {image.shape}")

        image = check_image_shape(image, logger)
        neuron = image[:,settings.neuron_channel-1,:,:]
        
    
        #load restoration model
        logger.info(f"Restoration model = {locations.rest_model_path}")
        """
        if settings.neuron_rest_type[0] == 'care':
            if os.path.isdir(neuron_rest_model_path) is False:
                raise RuntimeError(neuron_rest_model_path, "not found, check settings and model directory")
            rest_model = CARE(config=None, name=neuron_rest_model_path)
            neuron = image[:,settings.neuron_channel-1,:,:]
            logger.info(f"Section image shape: {neuron.shape}")
            #apply restoration model to channel
            #logger.info(f"Restoring image for channel {channel}")
            
            restored = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint16)
            
            start_time = time.time()
            for slice_idx in range(image.shape[0]):
                loop_start_time = time.time()
                
                logger.info(f"\rRestoring slice {slice_idx+1} of {image.shape[0]}") #, end="\r", flush=True)
                
                slice_img = image[slice_idx]
                with main.HiddenPrints():
                    
    
                    #restore image
                    restoredslice = rest_model.predict(slice_img, axes='YX', n_tiles=settings.tiles_for_prediction)
    
                    #convert to 16bit
                    restoredslice = restoredslice.astype(np.uint16)
                    #remove low intensities that are artifacts 
                    #as restored images have varying backgrounds due to high variability in samples. Detect background with median, then add the cutoff
                    #cutoff = np.median(restored) + rest_type[1]
                    #restored[restored < cutoff] = 0
                    background = restoration.rolling_ball(restoredslice, radius=5)
                    restoredslice = restoredslice - background
                    restored[slice_idx] = restoredslice
                    
                    loop_end_time = time.time()
                    loop_duration = loop_end_time - loop_start_time
                    total_elapsed_time = loop_end_time - start_time
                    avg_time_per_loop = total_elapsed_time / (slice_idx+1)
                    estimated_total_time = avg_time_per_loop * image.shape[0]

                    #logger.info(f"{loop_duration:.2f} seconds. Estimated total time: {estimated_total_time:.2f} minutes")
                    """
            #logger.info("Complete.\n")
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
            

def nnunet_create_labels(inputdir, settings, locations, logger):
    logger.info(f"\nDetecting spines and dendrites...")
    settings.shape_error = False
    settings.rescale_req = False
    
    #check if rescaling required and create scaling factors
    if settings.input_resZ != settings.neuron_seg_model_res[2] or settings.input_resXY != settings.neuron_seg_model_res[1]:
        logger.info(f" Images will be rescaled to match network.")
        
        settings.rescale_req = True
        #z in / z desired, y in / desired ...
        settings.scaling_factors = (settings.input_resZ/settings.neuron_seg_model_res[2],
                          settings.input_resXY/settings.neuron_seg_model_res[1], 
                          settings.input_resXY/settings.neuron_seg_model_res[0])
        #settings.inverse_scaling_factors = tuple(1/np.array(settings.scaling_factors))
        
        logger.info(f" Scaling factors: Z = {settings.scaling_factors[0]} Y = {settings.scaling_factors[1]} X = {settings.scaling_factors[2]} ") 
    
    #data can be raw data OR restored data so check channels
    files = [file_i
             for file_i in os.listdir(inputdir)
             if file_i.endswith('.tif')]
    files = sorted(files)
        
    #Prepare Raw data for nnUnet
    
    # Initialize reference to None - if using histogram matching
    logger.info(f" Histogram Matching is set to = {settings.HistMatch}")
    reference_image = None
    
    #create empty arrays to capture dims and padding info
    settings.original_shape = [None] * len(files)
    settings.padding_req = np.zeros(len(files))
    
    for file in range(len(files)):
        logger.info(f" Preparing image {file+1} of {len(files)} - {files[file]}")
        
        image = imread(inputdir + files[file])
        logger.info(f"  Raw data has shape: {image.shape}")
        
        image = check_image_shape(image, logger)
        
        neuron = image[:,settings.neuron_channel-1,:,:]
        

        
        # rescale if required by model        
        if settings.input_resZ != settings.neuron_seg_model_res[2] or settings.input_resXY != settings.neuron_seg_model_res[1]:
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
                                locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, settings)
    
    #logger.info(cmd)
    
    ##uncomment if issues with nnUnet
    #result = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, locations,settings)
    
    '''
    # Add environment to the system path
    #os.environ["PATH"] = settings.nnUnet_env_path + os.pathsep + os.environ["PATH"]

    #python = settings.nnUnet_env_path+"/python.exe"

    #result = subprocess.run(['nnUNetv2_predict', '-i', locations.nnUnet_input, '-o', locations.labels, '-d', dataset_id, '-c', settings.nnUnet_type, '--save_probabilities', '-f', 'all'], capture_output=True, text=True)
    #command = 'nnUNetv2_predict -i ' + locations.nnUnet_input + ' -o ' + locations.labels + ' -d ' + dataset_id + ' -c ' + settings.nnUnet_type + ' --save_probabilities -f all']
    
    #command = [python, "-m", "nnUNetv2_predict", "-i", locations.nnUnet_input, "-o", locations.labels, "-d", dataset_id, "-c", settings.nnUnet_type, "--save_probabilities", "-f", "all"]
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
            
            image = imread(locations.nnUnet_input + files[file])
      
            #Unpad if padded
            if settings.padding_req[file] == 1:
                image = image[2:-2, :, :]
            
            
            # rescale labels back up if required      
            if settings.input_resZ != settings.neuron_seg_model_res[2] or settings.input_resXY != settings.neuron_seg_model_res[1]:
                #logger.info(f"orignal settings shape: {settings.original_shape[file]}")
                image = resize(image, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                #logger.info(f"image resized: {labels.shape}")
    
                tifffile.imwrite(locations.nnUnet_input + files[file], image.astype(np.uint8), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))
                
    
    #logger.info("Segmentation complete.\n")
    return stdout

def run_nnunet_predict(conda_dir, nnUnet_env, input_dir, output_dir, dataset_id, nnunet_type, settings):
    # Set environment variables
    
    initialize_nnUnet(settings)
    
    activate_env = fr"{conda_dir}\Scripts\activate.bat && set PATH={conda_dir}\envs\{nnUnet_env}\Scripts;%PATH%"

    # Define the command to be run
    cmd = "nnUNetv2_predict -i \"{}\" -o \"{}\" -d {} -c {} --save_probabilities -f all".format(input_dir, output_dir, dataset_id, nnunet_type)
    
       
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
    nnUNetv2_predict -i "{input_dir}" -o "{output_dir}" -d {dataset_id} -c {nnunet_type} --save_probabilities -f all"""
    
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
    #command = 
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result

def initialize_nnUnet(settings):
    os.environ['nnUNet_raw'] = settings.nnUnet_raw
    os.environ['nnUNet_preprocessed'] = settings.nnUnet_preprocessed
    nnUnet_results = settings.neuron_seg_model_path
    os.environ['nnUNet_results'] = nnUnet_results
    


def analyze_spines(settings, locations, log, logger):
    logger.info("\nAnalyzing spines...\n")
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
        raise RuntimeError("Lists are not of equal length.")
    
    spine_summary = pd.DataFrame()
    
    for file in range(len(files)):
        logger.info(f' Analyzing image {file+1} of {len(files)} \n Raw Image: {files[file]} & Label Image: {label_files[file]}')
        
        image = imread(locations.input_dir + files[file])
        labels = imread(locations.labels + label_files[file])
        
        logger.info(f" Raw shape: {image.shape} & Labels shape: {labels.shape}")
        
        #Unpad if padded
        if settings.padding_req[file] == 1:
            labels = labels[2:-2, :, :]

            #tifffile.imwrite(locations.labels + label_files[file], labels.astype(np.uint8), imagej=True, photometric='minisblack',
            #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
             #       resolution=(settings.input_resXY, settings.input_resXY))
        
        
        # rescale labels back up if required      
        if settings.input_resZ != settings.neuron_seg_model_res[2] or settings.input_resXY != settings.neuron_seg_model_res[1]:
            logger.info(f"orignal settings shape: {settings.original_shape[file]}")
            labels = resize(labels, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
            logger.info(f"labels resized: {labels.shape}")

            tifffile.imwrite(locations.labels + label_files[file], labels.astype(np.uint8), imagej=True, photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                    resolution=(settings.input_resXY, settings.input_resXY))
            

        
        image = check_image_shape(image, logger) 
        
        neuron = image[:,settings.neuron_channel-1,:,:]

        
        spines = (labels == 1)
        dendrites = (labels == 2)
        #soma = (image == 3).astype(np.uint8)
        
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
        logger.info(" Detecting spines...")
        spine_labels = spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
    
    
        #Measurements
        spine_table, summary, spines_filtered = spine_measurements(image, spine_labels, settings.neuron_channel, dendrite_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, files[file], logger)
                                                          #soma_mask, soma_distance, )
        
        # update summary with additional metrics
        summary.insert(1, 'res_XY', settings.input_resXY)  
        summary.insert(2, 'res_Z', settings.input_resZ)
        dendrite_length = np.sum(skeleton == 1)
        summary.insert(3, 'dendrite_length', dendrite_length)
        dendrite_length_um = dendrite_length*settings.input_resXY
        summary.insert(4, 'dendrite_length_um', dendrite_length_um)
        dendrite_volume = np.sum(dendrites ==1)
        summary.insert(5, 'dendrite_vol', dendrite_volume)
        dendrite_volume_um3 = dendrite_volume*settings.input_resXY*settings.input_resXY*settings.input_resZ
        summary.insert(6, 'dendrite_vol_um3', dendrite_volume_um3)
        summary.insert(9, 'spines_per_um_length', summary['total_spines'][0]/dendrite_length_um)
        summary.insert(10, 'spines_per_um3_vol', summary['total_spines'][0]/dendrite_volume_um3)
                                                          
        #append to summary
        # Append to the overall summary DataFrame
        spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
        
        
        
        neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, spines_filtered, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        
 
        #Extract MIPs for each spine
        spine_MIPs, spine_slices, filtered_spine_MIPs, filtered_spine_slices= create_spine_arrays_in_blocks(image, spines_filtered, spine_table, settings.spine_roi_volume_size, settings, locations, files[file],  logger, settings.GPU_block_size)
    
    
        logger.info(f" Image processing complete for file {files[file]}\n")
    if settings.shape_error == True:
        logger.info(f"!!! One or more images moved to \\Not_Processed due to having\nless than 5 Z slices. Please modify these files before reprocessing.\n")
    spine_summary.to_csv(locations.tables + 'Detected_spines_summary.csv',index=False) 
    #logger.info("Spine analysis complete.\n")
    


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
    
    
    #! Fix this for multichannel raw data
    neuron = image
    
    
    spines = (labels == 1)
    dendrites = (labels == 2)
    
    spine_labels = spine_detection_4d(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #value used to remove small holes
    
    
    #create lists for each 3D volume
    dendrite_distance_list = []
    skeleton_list = []
        
    spine_summary = pd.DataFrame()
    
    for t in range(labels.shape[0]):
        
        logger.info(f" Processing timepoint {t+1} of {labels.shape[0]}.")
        
        labels_3d = labels[t, :, :, :]
    
    
        #neuron = image[:,settings.neuron_channel-1,:,:]

        #soma = (image == 3).astype(np.uint8)
        
        #Create Distance Map
        
        dendrites_3d = dendrites[t,:,:,:]
    
        dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites_3d)) #invert neuron mask to get outside distance  
        dendrite_distance_list.append(dendrite_distance)
        
        dendrites_3d = dendrites_3d.astype(np.uint8)
        skeleton = morphology.skeletonize_3d(dendrites_3d)
        skeleton_list.append(skeleton)
        #if settings.save_val_data == True:
        #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)
        
        #Create Neuron MIP for validation - include distance map too
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
    
        #Detection
        #logger.info(" Detecting spines...")
        
    dendrite_distance = np.stack(dendrite_distance_list, axis=0)
    skeleton = np.stack(skeleton_list, axis=0)
    
    spines_filtered_list = []
    all_spines_table = pd.DataFrame()
    all_summary_table = pd.DataFrame()
    
    #Measurements
    
    
    for t in range(labels.shape[0]):
        logger.info(f" Measuring timepoint {t+1} of {labels.shape[0]}.")
        spine_table, summary, spines_filtered = spine_measurements(image[t,:,:,:], spine_labels[t,:,:,:], settings.neuron_channel, dendrite_distance[t,:,:,:], settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, datasetname, logger)
                                                              #soma_mask, soma_distance, )
        if t == 0:
            previous_spines = set(np.unique(spines_filtered))
            new = 0
            pruned = 0
        
        else:
            current_spines = set(np.unique(spines_filtered))
            new = len(current_spines - previous_spines)
            pruned = len(previous_spines - current_spines)
        
        #add timepoint columns
        spine_table['timepoint'] = t
        summary['timepoint'] = t
        summary['new_spines'] = new
        summary['pruned_spines'] = pruned
        
        summary.insert(1, 'res_XY', settings.input_resXY)  
        summary.insert(2, 'res_Z', settings.input_resZ)
        dendrite_length = np.sum(skeleton == 1)
        summary.insert(3, 'dendrite_length', dendrite_length)
        dendrite_length_um = dendrite_length*settings.input_resXY
        summary.insert(4, 'dendrite_length_um', dendrite_length_um)
        dendrite_volume = np.sum(dendrites ==1)
        summary.insert(5, 'dendrite_vol', dendrite_volume)
        dendrite_volume_um3 = dendrite_volume*settings.input_resXY*settings.input_resXY*settings.input_resZ
        summary.insert(6, 'dendrite_vol_um3', dendrite_volume_um3)
        summary.insert(9, 'spines_per_um_length', summary['total_spines'][0]/dendrite_length_um)
        summary.insert(10, 'spines_per_um3_vol', summary['total_spines'][0]/dendrite_volume_um3)
        
        #append tables
        all_spines_table = pd.concat([all_spines_table, spine_table], ignore_index=True)
        all_summary_table = pd.concat([all_summary_table, summary], ignore_index=True)
        
        #append images
        spines_filtered_list.append(spines_filtered)
        
    
    spines_filtered_all = np.stack(spines_filtered_list, axis=0)
    
    col = all_summary_table.pop('new_spines')  
    all_summary_table.insert(9, 'new_spines', col) 
    col = all_summary_table.pop('pruned_spines')
    all_summary_table.insert(10, 'pruned_spines', col) 
    
    all_summary_table.rename(columns={'filename': 'dataset'}, inplace=True)
    all_spines_table.rename(columns={'filename': 'dataset'}, inplace=True)
    
    
    # update summary with additional metrics
    
                                                      
    #append to summary
    # Append to the overall summary DataFrame
    #spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
     
    #Create Pivot Tables for volume and coordinates
    
    vol_over_t = all_spines_table.pivot(index='label', columns='timepoint', values='spine_vol_um3')
    z_over_t = all_spines_table.pivot(index='label', columns='timepoint', values='z')
    y_over_t = all_spines_table.pivot(index='label', columns='timepoint', values='y')
    x_over_t = all_spines_table.pivot(index='label', columns='timepoint', values='x')
    
    #save required tables
    vol_over_t.to_csv(locations.tables + 'Volume_4D.csv',index=False) 
    all_summary_table.to_csv(locations.tables + 'Detected_spines_4D_summary.csv',index=False) 
    all_spines_table.to_csv(locations.tables + 'Detected_spines_4D.csv',index=False) 
    
    
    # Create MIP
    neuron_MIP = create_mip_and_save_multichannel_tiff_4d([neuron, spines, spines_filtered_all, dendrites, skeleton, dendrite_distance], locations.input_dir+"/Registered/Registered_MIPs_4D.tif", 'float', settings)
    
    #Extract MIPs for each spine
    spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.spine_roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

    #Cleanup
    if os.path.exists(locations.nnUnet_input): shutil.rmtree(locations.nnUnet_input)
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

def dendrite_detection(dendrite, logger):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    spines_clean = morphology.remove_small_holes(spines, holes)
    
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

def spine_measurements(image, labels, neuron_ch, dendrite_distance, sizes, dist, settings, locations, filename, logger):
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
    logger.info("  Measuring channel 1...")
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
        logger.info(f"  Measuring channel {ch+2}...")
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
    logger.info("  Measuring distances to dendrite...")
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
    


    #filter out small objects
    volume_min = sizes[0] #3
    volume_max = sizes[1] #1500?
    
    #logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")
    

    #filter based on volume
    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max) ] 
    
    #filter based on distance to dendrite
    #logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    
    filtered_table = filtered_table[(filtered_table['dist_to_dendrite'] < dist)] 
    
    #create vol um measurement
    filtered_table.insert(5, 'spine_vol_um3', filtered_table['area'] * (settings.input_resXY*settings.input_resXY*settings.input_resZ))
    filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)
    
    #create dist um cols
    filtered_table.insert(9, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY*settings.input_resXY))
    
    filtered_table.insert(11, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY*settings.input_resXY))
    
    
    labels = create_filtered_labels_image(labels, filtered_table)

    logger.info(f" After filtering {len(filtered_table)} spines remain from total of {len(main_table)}")

    #create summary table
    
        
    spine_reduced = filtered_table.drop(columns=['label', 'z', 'y', 'x'])

    # Generate summary
    summary_stats = spine_reduced.mean().to_dict()
    summary_stats['avg_spine_vol'] = summary_stats.pop('spine_vol')  # Rename 'area' to 'Volume'
    summary_stats['avg_spine_vol_um3'] = summary_stats.pop('spine_vol_um3')  # Rename 'area' to 'Volume'
    summary_stats['avg_dist_to_dendrite'] = summary_stats.pop('dist_to_dendrite') 
    summary_stats['avg_spine_length'] = summary_stats.pop('spine_length') 
    summary_stats['avg_dist_to_dendrite_um'] = summary_stats.pop('dist_to_dendrite_um') 
    summary_stats['avg_spine_length_um'] = summary_stats.pop('spine_length_um') 
    
    summary_stats['avg_C1_mean_int'] = summary_stats.pop('C1_mean_int')  
    summary_stats['avg_C1_max_int'] = summary_stats.pop('C1_max_int')  

    for ch in range(image.shape[1]-1):
        logger.info(f" Measuring channel {(ch+2)}...")
        summary_stats['avg_C'+str(ch+2)+'_mean_int'] = summary_stats.pop('C'+str(ch+2)+'_mean_int')  # Rename 'area' to 'Volume'
        summary_stats['avg_C'+str(ch+2)+'_max_int'] = summary_stats.pop('C'+str(ch+2)+'_max_int')  # Rename 'area' to 'Volume'
    
    #shorten filename
    filename = filename.replace('.tif', '')

    # Convert summary to a DataFrame
    summary_df = pd.DataFrame(summary_stats, index=[0])
    summary_df.insert(0, 'total_spines', main_table.shape[0])
    summary_df.insert(1, 'total_filtered_spines', filtered_table.shape[0])
    summary_df.insert(0, 'Filename', filename)  # Insert a column at the beginning


    filtered_table.to_csv(locations.tables + 'Detected_spines_'+filename+'.csv',index=False) 
  
    return filtered_table, summary_df, labels



def create_filtered_labels_image(labels, filtered_table):
    """
    Create a new labels image using a filtered regionprops table, more efficiently.

    Args:
        labels (numpy.ndarray): Input labels image.
        filtered_table (pd.DataFrame): Filtered regionprops table.

    Returns:
        numpy.ndarray: Filtered labels image.
    """
    # Create a mask that retains only the labels present in the filtered_table
    mask = np.isin(labels, filtered_table['label'].values)

    # Create the filtered labels image by multiplying the mask with the original labels
    filtered_labels = labels * mask

    return filtered_labels

def create_filtered_and_unfiltered_spine_arrays_cupy(image, labels_filtered, table, volume_size, settings, locations, file):
    mip_list = []
    slice_z_list = []
    mip_label_filtered_list = []
    slice_before_mip_list = []
    
    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)
    
    image_cp = cp.array(image)
    labels_filtered_cp = cp.array(labels_filtered)
    
    volume_size_z = int(volume_size / settings.input_resZ)
    volume_size_y = int(volume_size / settings.input_resXY)
    volume_size_x = int(volume_size / settings.input_resXY)


    # Pad the image and labels_filtered and keep the channel axis intact for the image
    image_cp = cp.pad(image_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2), (0, 0)), mode='constant', constant_values=0)
    labels_filtered_cp = cp.pad(labels_filtered_cp, ((volume_size_z // 2, volume_size_z // 2), (volume_size_y // 2, volume_size_y // 2), (volume_size_x // 2, volume_size_x // 2)), mode='constant', constant_values=0)


    for index, row in table.iterrows():
        #logger.info(f' Extracting and saving 3D ROIs for cell {index + 1}/{len(table)}', end='\r')
        z, y, x, label = int(row['z']), int(row['y']), int(row['x']), int(row['label'])
        
        z_min, z_max = z - volume_size_z // 2, z + volume_size_z // 2
        y_min, y_max = y - volume_size_y // 2, y + volume_size_y // 2
        x_min, x_max = x - volume_size_x // 2, x + volume_size_x // 2
        
        # Extract the volume and keep the channel axis intact
        extracted_volume = image_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2, :]
        extracted_labels = labels_filtered_cp[z_min + volume_size_z // 2:z_max + volume_size_z // 2, y_min + volume_size_y // 2:y_max + volume_size_y // 2, x_min + volume_size_x // 2:x_max + volume_size_x // 2]

        # Extract 2D slice at position z
        slice_z = extracted_volume[volume_size_z // 2, :, :, :]

        # Filter the volume to only show image data inside the label
        label_mask = (extracted_labels == label)
        label_mask_expanded = cp.expand_dims(label_mask, axis=-1)
        extracted_volume_label_filtered = extracted_volume * label_mask_expanded

        # Extract 2D slice before the MIP in step 3 is created
        slice_before_mip = extracted_volume_label_filtered[volume_size_z // 2, :, :, :]

        # Compute MIPs
        mip = cp.max(extracted_volume, axis=0)
        mip_label_filtered = cp.max(extracted_volume_label_filtered, axis=0)

        mip_list.append(mip.get())
        slice_z_list.append(slice_z.get())
        mip_label_filtered_list.append(mip_label_filtered.get())
        slice_before_mip_list.append(slice_before_mip.get())
        
        del mip, slice_z, mip_label_filtered, slice_before_mip, extracted_volume, extracted_labels
        cp.cuda.Stream.null.synchronize()
        gc.collect()

    mip_array = np.stack(mip_list)
    mip_array = np.moveaxis(mip_array, 3, 1)
    
    slice_z_array = np.stack(slice_z_list)
    slice_z_array = np.moveaxis(slice_z_array, 3, 1)
    
    mip_label_filtered_array = np.stack(mip_label_filtered_list)
    mip_label_filtered_array = np.moveaxis(mip_label_filtered_array, 3, 1)
    
    slice_before_mip_array = np.stack(slice_before_mip_list)
    slice_before_mip_array = np.moveaxis(slice_before_mip_array, 3, 1)
    
    return mip_array, slice_z_array, mip_label_filtered_array, slice_before_mip_array

def create_spine_arrays_in_blocks(image, labels_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
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
                        block_image, block_labels_filtered, block_table, volume_size, settings, locations, file
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
                        block_image, block_labels_filtered, block_table, volume_size, settings, locations, file
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
    #spine_MIPs, filtered_spine_MIP = create_MIP_spine_arrays_in_blocks_4d(neuron_MIP, y_over_t, x_over_t, settings.spine_roi_volume_size, settings, locations, datasetname, logger, settings.GPU_block_size)

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