# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""
__title__     = 'spinpipe'
__version__   = '0.9.1'
__date__      = "25 July, 2023"
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
from skimage.io import imread #, util #imsave, imshow,

#from math import trunc

import skimage.io
#from skimage.transform import rescale
from skimage import measure, morphology, segmentation, exposure #util,  color, data, filters,  exposure, restoration 
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
    logger.info(f"\nCreating dendrite and spine masks...\n")
    
    #data can be raw data OR restored data so check channels
    files = [file_i
             for file_i in os.listdir(inputdir)
             if file_i.endswith('.tif')]
    files = sorted(files)
        
    #Prepare Raw data for nnUnet
    
    # Initialize reference to None - if using histogram matching
    logger.info(f"Histogram Matching is set to = {settings.HistMatch}")
    reference_image = None
    
    for file in range(len(files)):
        logger.info(f" Preparing images {file+1} of {len(files)} - {files[file]}")

        image = imread(inputdir + files[file])
        logger.info(f" Raw data has shape {image.shape}")
        
        image = check_image_shape(image, logger)
        
        neuron = image[:,settings.neuron_channel-1,:,:]
        
        if settings.HistMatch == True:
            # If reference_image is None, it's the first image.
            if reference_image is None:
                neuron = contrast_stretch(neuron, pmin=0, pmax=100)
                reference_image = neuron
            else:
               # Match histogram of current image to the reference image
               neuron = exposure.match_histograms(neuron, reference_image)    
            
        
        #save neuron as a tif file in nnUnet_input - if file doesn't end with 0000 add that at the end
        name, ext = os.path.splitext(files[file])

        if not name.endswith("0000"):
            name += "_0000"

        new_filename = name + ext
        
        filepath = locations.nnUnet_input+new_filename
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(filepath, neuron.astype(np.uint16), plugin='tifffile', photometric='minisblack')
            
    #Run nnUnet over prepared files
    #initialize_nnUnet(settings)
    
    # split the path into subdirectories
    subdirectories = os.path.normpath(settings.neuron_seg_model_path).split(os.sep)
    last_subdirectory = subdirectories[-1]
    # find all three digit sequences in the last subdirectory
    matches = re.findall(r'\d{3}', last_subdirectory)
    # If there's a match, assign it to a variable
    dataset_id = matches[0] if matches else None

    
    logger.info("\nPerforming U-Net segmentation of neurons...")
    
    
    #logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")
    
    stdout = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, 
                                locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, settings)
    
    
    
    
    #result = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, locations,settings)
    
    
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
    logger.info(stdout)
    #delete nnunet input folder and files
    
    #if os.path.exists(locations.nnUnet_input):
        #shutil.rmtree(locations.nnUnet_input)
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
    
    logger.info("Segmentation complete.\n")
    return stdout

def run_nnunet_predict(conda_dir, nnUnet_env, input_dir, output_dir, dataset_id, nnunet_type, settings):
    # Set environment variables
    
    initialize_nnUnet(settings)
    
    activate_env = fr"{conda_dir}\Scripts\activate.bat && set PATH={conda_dir}\envs\{nnUnet_env}\Scripts;%PATH%"

    # Define the command to be run
    cmd = "nnUNetv2_predict -i {} -o {} -d {} -c {} --save_probabilities -f all".format(input_dir, output_dir, dataset_id, nnunet_type)

       
    # Combine the activate environment command with the actual command
    final_cmd = f'{activate_env} && {cmd}'

    # Run the command
    process = subprocess.Popen(final_cmd, shell=True)
    stdout, stderr = process.communicate()
    return stdout
    
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
    os.environ['nnUNet_results'] = settings.nnUnet_results
    


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
        raise RuntimeError("Lists are not of equal length.")
    
    spine_summary = pd.DataFrame()
    
    for file in range(len(files)):
        logger.info(f' Analyzing image {file+1} of {len(files)} \n  Raw Image:{files[file]} \n  Label Image:{label_files[file]}')
        
        image = imread(locations.input_dir + files[file])
        labels = imread(locations.labels + label_files[file])
        
        logger.info(f"  Raw data has shape {image.shape}")
        
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
        spine_labels = spine_detection(spines, 10 ** 3, logger) #value used to remove small holes
    
    
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
    
    
        logger.info(f"\nImage processing complete for file {files[file]}\n")
    spine_summary.to_csv(locations.tables + 'Detected_spines_summary.csv',index=False) 
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
def spine_detection(spines, holes, logger):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spines_clean = morphology.remove_small_holes(spines, holes)
    
    spines_clean = label(spines_clean)
    #immune_labels_full = segmentation.watershed(immune_all_clean, immune_labels, mask=immune_all_clean)
    
    #Remove objects touching border
    padded = np.pad(
      spines_clean,
      ((1, 1), (0, 0), (0, 0)),
      mode='constant',
      constant_values=0,
      )
    spines_clean = segmentation.clear_border(padded)[1:-1]

    return spines_clean


def spine_measurements(image, labels, neuron_ch, dendrite_distance, sizes, dist, settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)
        
    Returns:
        pandas table and labeled spine image
    """


    #Measure channel 1:
    logger.info(" Measuring channel 1...")
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
        logger.info(f" Measuring channel {ch+2}...")
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
    logger.info(" Measuring distance to dendrite...")
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
    
    logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")
    

    #filter based on volume
    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max) ] 
    
    #filter based on distance to dendrite
    logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    
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

    logger.info(f'Total number of blocks used for GPU spine array calculations: {total_blocks} ')

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

def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))