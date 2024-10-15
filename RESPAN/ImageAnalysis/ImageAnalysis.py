# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""

__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'GPL-3.0 License (see LICENSE)'
__copyright__ = 'Copyright © 2024 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'

import multiprocessing
import time
import os
import numpy as np
import pandas as pd
import math
import warnings
import re
import shutil
import sys
import gc

import memory_profiler
import psutil

import contextlib
from pathlib import Path
import subprocess
import threading
from collections import defaultdict

import trimesh
import pyvista as pv
from multiprocessing import Pool
from cupyx.scipy import ndimage as cp_ndimage
from collections import deque
from scipy.ndimage import distance_transform_edt, gaussian_filter
import cupy as cp
from skimage.graph import route_through_array
from tifffile import imread, imwrite
from skimage import measure, morphology, segmentation, exposure, graph #util,  color, data, filters,  exposure, restoration
from skimage.transform import resize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import generate_binary_structure
from scipy import ndimage
from csbdeep.models import CARE
from cupyx.scipy.ndimage import binary_dilation
from scipy.spatial import distance
from scipy.interpolate import splprep, splev

import RESPAN.Main.Main as main
import cupy.cuda.runtime as rt
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit
from skimage.measure import marching_cubes
from cupyx.scipy.ndimage import label as cp_label
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.csgraph import connected_components
from skimage.graph import MCP_Geometric

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
        data = locations.restored

    if  settings.image_restore == False and settings.axial_restore == True:
        #restore axial resolution from raw data
        axial_restore_image(locations.input_dir, settings, locations, logger)
        data = locations.input_dir + '/selfnet/'


    elif settings.image_restore == True and settings.axial_restore == True:
        #restore axial resolution on CARE restored data
        axial_restore_image(data, settings, locations, logger)
        data = locations.input_dir + '/selfnet/'

    else:
        data = locations.input_dir

    #create nnunet labels
        # #include options here for alternative unets if required
    #logger.info(f"Imnporting from dir {data}")
    log = nnunet_create_labels(data, settings, locations, logger)

    return log

def axial_restore_image(inputdir, settings, locations, logger):
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Performing SelfNet axial restoration neuron channel...")
    logger.info("   ** Be aware that SelfNet restoration greatly increases file size (up to 10x) **")
    logger.info("   As a result, storage requirements must account for this, and processing time will be increased accordingly.")
    #SelfNet Inference
    # nnunet_env = 'nnunet' # may need to provide conda dir for max compat
    # model path should be the most recent file in saved models checkpoint
    # update the training file to take this file and place it somewhere safe
    # e.g. model_path=input_dir+'checkpoint/saved_models/deblur_net_60_3200.pkl'

    # find most recent pkl file in model_dir and update that to be the model path
    if settings.selfnet_path != None:

        #parser.add_argument('--input_dir', type=str, required=True, help='The input directory')
        #parser.add_argument('--neuron_ch', type=int, default=0, help='the channel for inference')
        #parser.add_argument('--model_path', type=str, required=True, help='The model path')
        #parser.add_argument('--min_v', type=int, default=0, help='The minimum intensity')
        #parser.add_argument('--max_v', type=int, default=65535, help='The maximum intensity')
        #parser.add_argument('--scale', type=float, default=0.21, help='The resolution scaling factor') XY sampling / z step e.g. 75/150 = 0.5
        #parser.add_argument('--z_step', type=int, default=1, help='The final z-resolution')

        args_dict = {
            'input_dir': inputdir,
            'neuron_ch': settings.neuron_channel-1,
            'model_path': settings.selfnet_path,
            'min_v': 0,
            'max_v': 65535,
            'scale':  settings.input_resXY,
            'z_step': settings.input_resZ
        }
        #logger info all args_dict

        logger.info(f"\n   Input xy-resolution: {settings.input_resXY}. Input Z resolution: { settings.input_resZ}")
        logger.info(f"   Final z-resolution: {settings.input_resXY}")

        run_external_script(
            settings.basepath + "\SelfNet_Inference.py",
            settings.nnUnet_env, args_dict)


        #update resolution for remaing calculations
        settings.input_resZ = settings.input_resXY
        logger.info(
            "Restoration complete.")
        #logger.info(settings.input_resZ)

    else:
        logger.info("SelfNet model path not found in settings file, update settings file - skipping axial restoration.")

    logger.info(
       "-----------------------------------------------------------------------------------------------------")


def run_external_script(script_path, conda_env, args_dict):
    args_str = ' '.join(f'--{k} "{v}"' for k, v in args_dict.items())
    command = f'conda run -n {conda_env} python "{script_path}" {args_str}'
    process = subprocess.Popen(command, shell=True)
    process.wait()


def selfnet_inference(script_path, respan_env, model_dir, min_v, max_v, scale, z_step, settings):
    #nnunet_env = 'nnunet' # may need to provide conda dir for max compat
    #model path should be the most recent file in saved models checkpoint
    #update the training file to take this file and place it somewhere safe
    #e.g. model_path=input_dir+'checkpoint/saved_models/deblur_net_60_3200.pkl'

    #find most recent pkl file in model_dir and update that to be the model path
    args_dict = {
        'input_dir': input,
        'model_path': model_dir,
        'min_v': min_v,
        'max_v': max_v,
        'scale': scale,
        'z_step': z_step
    }
    run_external_script(
                    settings.basepath+"\SelfNet_Inference.py",
                    settings.nnUnet_env, args_dict)


def restore_image (inputdir, settings, locations, logger):
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Restoring images with CARE models...")

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

                #find min int above zero and use this to replace all zero values
                min_intensity = np.min(channel_image[np.nonzero(channel_image)])

                channel_image = np.where(channel_image == 0, min_intensity, channel_image)

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
                imwrite(locations.restored+ files[file], restored, compression=('zlib', 1), imagej=True, photometric='minisblack',
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
    logger.info(f" Detecting spines and dendrites...")
    settings.shape_error = False
    settings.rescale_req = False

    #check if rescaling required and create scaling factors
    if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
        logger.info(f"  Images will be rescaled to match network.")

        settings.rescale_req = True
        #z in / z desired, y in / desired ...
        settings.scaling_factors = (settings.input_resZ/settings.model_resZ,
                                    settings.input_resXY/settings.model_resXY,
                                    settings.input_resXY/settings.model_resXY)
        #settings.inverse_scaling_factors = tuple(1/np.array(settings.scaling_factors))

        logger.info(f"  Scaling factors: Z = {round(settings.scaling_factors[0],2)} Y = {round(settings.scaling_factors[1],2)} X = {round(settings.scaling_factors[2],2)} ")

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
        logger.info(f"  *Spines and dendrites already detected. \nDelete \Validation_Data\Segmentation_Labels if you wish to regenerate.")
        stdout = None
        settings.prev_labels = True

        return_code = 0
    else:

        #Prepare Raw data for nnUnet

        # Initialize reference to None - if using histogram matching
        #logger.info(f" Histogram Matching is set to = {settings.HistMatch}")
        reference_image = None
        settings.prev_labels = False

        for file in range(len(files)):
            logger.info(f"   Preparing image {file+1} of {len(files)} - {files[file]}")

            image = imread(inputdir + files[file])
            logger.info(f"   Raw data has shape: {image.shape}")

            image = check_image_shape(image, logger)

            if settings.axial_restore == True:
                neuron = image[:,0,:,:]
            else:
                neuron = image[:,settings.neuron_channel-1,:,:]

            # rescale if required by model        
            if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                settings.original_shape[file] = neuron.shape
                #new_shape = (int(neuron.shape[0] * settings.scaling_factors[0]), neuron.shape[1] * settings.scaling_factors[1]), neuron.shape[2] * settings.scaling_factors[2]))
                new_shape = tuple(int(dim * factor) for dim, factor in zip(neuron.shape, settings.scaling_factors))
                neuron = resize(neuron, new_shape, mode='constant', preserve_range=True, anti_aliasing=True)
                logger.info(f"   Data rescaled to match model for labeling has shape: {neuron.shape}")

            # logger.info the  it will take for processing image by dividing the number of pixels by a scaling factor
            #limit variable to 2 decimal places

            logger.info(f"    Estimated time to detect spines and dendrites for this image: {round(neuron.size * 2.5e-8,2)} minutes.\n")
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
                logger.info(f"   Too few Z-slices, padding to allow analysis.")

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
                imwrite(filepath, neuron.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY), )


        #Run nnUnet over prepared files
        #initialize_nnUnet(settings)

        # split the path into subdirectories
        subdirectories = os.path.normpath(settings.neuron_seg_model_path).split(os.sep)
        last_subdirectory = subdirectories[-1]
        # find all three digit sequences in the last subdirectory
        matches = re.findall(r'\d{3}', last_subdirectory)
        # If there's a match, assign it to a variable
        dataset_id = matches[0] if matches else None


        logger.info("\n  Performing spine and dendrite detection on GPU...")

        ##uncomment if issues with nnUnet
        #logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")




        return_code = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env,
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

        files = [file_i
                 for file_i in os.listdir(locations.labels)
                 if file_i.endswith('.tif')]
        files = sorted(files)

        for file in range(len(files)):
            #if file == 0: logger.info(' Unpadding and rescaling neuron channel for registration and time tracking...')

            # Unpad if padded # later update - these can be included in Unet processing stage to simplify!
            if settings.padding_req[file] == 1: #and settings.prev_labels == False and settings.original_shape[0] != None:
                logger.info(f"  Unpadding image {files[file]}")
                image = imread(locations.labels + files[file])
                image = image[2:-2, :, :]

                logger.info(f"  Image {files[file]} has shape: {image.shape}")

                imwrite(locations.labels + files[file], image.astype(np.uint8), compression=('zlib', 1), imagej=True,
                        photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))

                logger.info(f"  Image {files[file]} has been unpaded and saved to {locations.labels}")

                #Unpad if padded
                #if settings.padding_req[file] == 1:
                #    image = image[2:-2, :, :]
                #image = imread(locations.nnUnet_input + files[file])
                #logger.info(f"  Image {files[file]} has shape: {image.shape}")



                # rescale labels back up if required
                #if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                    #logger.info(f"orignal settings shape: {settings.original_shape[file]}")
                #    image = resize(image, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                    #logger.info(f"image resized: {labels.shape}")

                #    imwrite(locations.nnUnet_input + files[file], image.astype(np.uint8), imagej=True, photometric='minisblack',
                 #           metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                  #          resolution=(settings.input_resXY, settings.input_resXY))


    #logger.info("Segmentation complete.\n")
    logger.info("\n-----------------------------------------------------------------------------------------------------")
    return return_code

def run_nnunet_predict(conda_dir, nnUnet_env, input_dir, output_dir, dataset_id, nnunet_type, settings, logger):
    # Set environment variables

    initialize_nnUnet(settings, logger)

    activate_env = fr"{conda_dir}\Scripts\activate.bat {nnUnet_env }&& set PATH={settings.nnUnet_env_path}/{nnUnet_env}/Scripts;%PATH%"

    # Define the command to be run
    #cmd = "nnUNetv2_predictRESPAN -i \"{}\" -o \"{}\" -d {} -c {} -f all".format(input_dir, output_dir, dataset_id, nnunet_type)
    cmd = "nnUNetv2_predict -i \"{}\" -o \"{}\" -d {} -c {} -f all".format(input_dir, output_dir, dataset_id,
                                                                                 nnunet_type)

    # Combine the activate environment command with the actual command
    final_cmd = f'{activate_env} && {cmd}'

    return_code, stdout_out, stderr_out = run_process_with_logging(final_cmd, logger)
    # Run the command
    if return_code != 0:
        logger.info(f"Error: Command failed with return code {return_code}")
        if stderr_out:
            logger.info(f"Error message:\n{stderr_out}")
        if stdout_out:
            logger.info(f"Output message:\n{stdout_out}")

    #process = subprocess.Popen(final_cmd, shell=True)
    #stdout, stderr = process.communicate()
    return return_code


'''
def log_output(pipe, logger, buffer):
    for line in iter(pipe.readline, '' if isinstance(pipe, io.TextIOWrapper) else b''):
        if isinstance(line, bytes):
            decoded_line = line.decode().strip()
        else:
            decoded_line = line.strip()

        # Append all output to the buffer
        buffer.append(decoded_line + '\n')

        if (decoded_line.lower().startswith(("predicting", "done")) or
                re.match(r'^\d+', decoded_line)):
            logger.info(" " + decoded_line)
            print(decoded_line, flush=True)  # Print to terminal in real-time
'''


def clean_percentage_line(line):
    # Extract percentage and time
    match = re.search(r'(\d+)%.*?([\d:]+)', line)
    if match:
        percentage, time = match.groups()
        return f"{percentage}% [Time left: {time}]"
    return line


def log_output(pipe, logger, buffer):
    while True:
        try:
            line = pipe.readline()
            if not line:
                break

            if isinstance(line, bytes):
                try:
                    decoded_line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    decoded_line = line.decode('latin-1', errors='replace').strip()
            else:
                decoded_line = line.strip()

            # Append all output to the buffer
            buffer.append(decoded_line + '\n')
            print50 = 0
            # Clean and filter the output
            if "%" in decoded_line:
                cleaned_line = clean_percentage_line(decoded_line)
                if "50%" in cleaned_line and print50 == 0:# or "100%" in cleaned_line:
                    logger.info(f"   {cleaned_line}")
                    print(cleaned_line, flush=True)
                    print50 = 1
            elif decoded_line.lower().startswith(("predicting", "done")) or re.match(r'^\d+', decoded_line):
                logger.info(f"   {decoded_line}")
                print(decoded_line, flush=True)

        except Exception as e:
            print("")
            #print(f"Error processing line: {str(e)}", flush=True)
            #logger.error(f"Error processing line: {str(e)}")

def run_process_with_logging(cmd, logger):
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) #buffsize=1
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=1,
                               universal_newlines=True)

    stdout_buffer = []
    stderr_buffer = []

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logger, stdout_buffer))
    stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logger, stderr_buffer))

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to complete
    process.wait()

    stdout_thread.join()
    stderr_thread.join()

    # Wait for the output threads to finish
    stdout_output = ''.join(stdout_buffer)
    stderr_output = ''.join(stderr_buffer)

    return process.returncode, stdout_output, stderr_output   #process.stderr.read().decode().strip()


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
        #logger.info(log)
        raise RuntimeError("Number of raw and label images are not the same - check data.")

    spine_summary = pd.DataFrame()

    for file in range(len(files)):
        logger.info(f' Analyzing image {file+1} of {len(files)} \n  Raw Image: {files[file]} & Label Image: {label_files[file]}')

        settings.filename = files[file].replace(".tif", "")
        image = imread(locations.input_dir + files[file])
        labels = imread(locations.labels + files[file]) # use original file name to ensure correct image regardless of sorting

        logger.info(f"  Raw shape: {image.shape} & Labels shape: {labels.shape}")


        # if using axial resotration - image needs to be scaled to match the labels
        if settings.axial_restore == True:
            #rescale image to match labels
            if len(image.shape)==3:
                image = resize(image, labels.shape, order=0, mode='constant', preserve_range=True, anti_aliasing=True)
            elif len(image.shape)==4:
                # Broadcast the labels to have the same shape in axis 1 as the image
                labels_reshaped = labels[:, np.newaxis, ...]  # Add a new axis at position 1
                # Use np.tile to match the channel dimension of  image
                labels_reshaped = np.tile(labels_reshaped, (1, image.shape[1], 1, 1))

                # Resize the image to match the reshaped labels' shape
                image = resize(image, labels_reshaped.shape, order=0, mode='constant', preserve_range=True,
                               anti_aliasing=True)
                del labels_reshaped

            logger.info(f"  Due to axial restoration, image rescaled to match labels: {image.shape}")

        # rescale labels back up if required    

        if settings.axial_restore == False:
            if settings.original_shape[0] != None and settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY and settings.prev_labels == False:
                logger.info(f"  orignal settings shape: {settings.original_shape[file]}")
                labels = resize(labels, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                logger.info(f"  labels resized: {labels.shape}")

                imwrite(locations.labels + label_files[file], labels.astype(np.uint8), compression=('zlib', 1), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))


        #process images through Vaa3d
        if settings.Vaa3d == True and os.path.exists(settings.Vaa3Dpath):
            logger.info(f"  Creating SWC file using Vaa3D...")
            create_dir(locations.swcs)
            if os.path.exists(locations.swcs + files[file]+".swc"):
                logger.info(f"   {files[file]}.swc already exists, delete this file to regenerate.")
            else:
                # can no longer use                 vaa3D_neuron = (labels >= 2)*255 as necks are labeled with 4 and spines with 1
                # vaa3D_neuron = (labels >= 2)*255
                vaa3D_mask = np.where((labels >= 2) & (labels <= 3), 1, 0)
                neuron = image[:, settings.neuron_channel - 1, :, :]
                masked_neuron = neuron * vaa3D_mask

                imwrite(locations.swcs + files[file], masked_neuron.astype(np.uint8), compression=('zlib', 1), imagej=True, photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
                #run Vaa3D on image:
                cmd = '"{}" /x vn2 /f app2 /i "{}" /o "{}" /p NULL 0 1 1 1 1 0 5 1 0 0'.format(settings.Vaa3Dpath, locations.swcs + files[file], locations.swcs + files[file]+".swc ")
                #logger.info(cmd)
                # Run the command
                process = subprocess.Popen(cmd, shell=True)
                stdout, stderr = process.communicate()
                #logger.info(stdout + stderr)

                os.remove(locations.swcs + files[file])
                # Clear variables to free up memory
                del vaa3D_mask
                del neuron
                del masked_neuron
                # Force garbage collection
                gc.collect()

        image = check_image_shape(image, logger)

        #dendrite specific superceeded by whole neuron analysis
        #if settings.analysis_method == "Dendrite Specific":
        #    spine_summary = spine_and_dendrite_processing(image, labels, spine_summary, settings, locations, files[file], log, logger)
        #else:
        spine_summary = spine_and_whole_neuron_processing(image, labels, spine_summary, settings, locations, files[file], log, logger)


        if settings.shape_error == True:
            logger.info(f"!!! One or more images moved to \\Not_Processed due to having\nless than 5 Z slices. Please modify these files before reprocessing.\n")

        spine_summary.to_csv(locations.tables + 'Detected_spines_summary.csv',index=False)

    logger.info("RESPAN analysis complete.")

def associate_spines_with_necks_gpu_optimized(spines, necks, logger, max_overlap=25, batch=50):
    # Convert to CuPy arrays
    labeled_spines_cp = cp.array(spines)
    necks_cp = cp.array(necks > 0)

    # Create a structuring element (3x3x3 cube) for dilation
    struct_elem = cp.ones((3, 3, 3), dtype=cp.bool_)

    volume_shape = labeled_spines_cp.shape

    # Get bounding boxes for each labeled region
    props = measure.regionprops(cp.asnumpy(labeled_spines_cp))
    bboxes = [prop.bbox for prop in props]

    # Initialize the result array
    result = cp.zeros_like(labeled_spines_cp)

    # Process each labeled region
    for label_id, bbox in enumerate(bboxes, start=1):
        # Calculate the expansion amount for each dimension
        expansions = [
            min(max_overlap, bbox[0]),  # top
            min(max_overlap, bbox[1]),  # left
            min(max_overlap, bbox[2]),  # front
            min(max_overlap, volume_shape[0] - bbox[3]),  # bottom
            min(max_overlap, volume_shape[1] - bbox[4]),  # right
            min(max_overlap, volume_shape[2] - bbox[5])  # back
        ]

        # Expand the bounding box within limits
        expanded_bbox = [
            bbox[0] - expansions[0],
            bbox[1] - expansions[1],
            bbox[2] - expansions[2],
            bbox[3] + expansions[3],
            bbox[4] + expansions[4],
            bbox[5] + expansions[5]
        ]

        # Extract the region of interest
        spine_roi = labeled_spines_cp[expanded_bbox[0]:expanded_bbox[3],
                    expanded_bbox[1]:expanded_bbox[4],
                    expanded_bbox[2]:expanded_bbox[5]]
        neck_roi = necks_cp[expanded_bbox[0]:expanded_bbox[3],
                   expanded_bbox[1]:expanded_bbox[4],
                   expanded_bbox[2]:expanded_bbox[5]]

        # Process the region
        growth_occurred = True
        while growth_occurred:
            dilated_spine = binary_dilation(spine_roi == label_id, structure=struct_elem)
            new_growth = dilated_spine & neck_roi & (spine_roi == 0)
            spine_roi = cp.where(new_growth, label_id, spine_roi)
            growth_occurred = new_growth.any()

        # Update the result
        result[expanded_bbox[0]:expanded_bbox[3],
        expanded_bbox[1]:expanded_bbox[4],
        expanded_bbox[2]:expanded_bbox[5]] = spine_roi

        # Log the expansion for this spine
        logger.info(f"Processed spine {label_id} with expansions: {expansions}")

    return result.get()

def associate_spines_with_necks_gpu(spines, necks, logger):
    # Convert to CuPy arrays
    labeled_array_cp = cp.array(spines)
    binary_array_cp = cp.array( necks > 0)

    # Create a structuring element (3x3x3 cube) for dilation
    struct_elem = cp.ones((3, 3, 3), dtype=cp.bool_)

    # Initialize a flag to check if further growth is possible
    growth_occurred = True

    while growth_occurred:
        # Perform dilation on the labeled array using the structuring element
        dilated_labels_cp = binary_dilation(labeled_array_cp > 0, structure=struct_elem)

        # Find new growth locations where the dilation intersects with the binary array
        new_growth_cp = dilated_labels_cp & binary_array_cp & (labeled_array_cp == 0)

        # Get the label IDs of the neighboring regions (propagate the correct IDs)
        expanded_labels_cp = cp.zeros_like(labeled_array_cp)
        for label_id in cp.unique(labeled_array_cp):
            if label_id == 0:
                continue
            label_mask = (labeled_array_cp == label_id)
            dilated_label_mask = binary_dilation(label_mask, structure=struct_elem)
            expanded_labels_cp = cp.where(dilated_label_mask & new_growth_cp, label_id, expanded_labels_cp)

        # Apply the expanded labels to the labeled array
        labeled_array_cp = cp.where(new_growth_cp, expanded_labels_cp, labeled_array_cp)

        # Check if any growth occurred
        growth_occurred = new_growth_cp.any()

    # Return the final labeled array after fully growing
    return labeled_array_cp.get()


def get_bounding_box_cupy(mask, margin, shape, settings):
    """Get the bounding box of a binary mask with added margin."""
    margin_y = margin_x = int(margin / settings.input_resXY)
    margin_z = int(margin / settings.input_resZ)
    z, y, x = cp.where(mask)

    z_min, z_max = int(max(z.min().item() - margin_z, 0)), int(min(z.max().item() + margin_z + 1, shape[0]))
    y_min, y_max = int(max(y.min().item() - margin_y, 0)), int(min(y.max().item() + margin_y + 1, shape[1]))
    x_min, x_max = int(max(x.min().item() - margin_x, 0)), int(min(x.max().item() + margin_x + 1, shape[2]))

    return z_min, z_max, y_min, y_max, x_min, x_max


def pathfinding_v3b(object_subvolume_gpu, target_subvolume_gpu, intensity_image_gpu, logger):
    """
    Pathfinding that combines distance map and intensity image to prioritize brighter voxels,
    with optimized start and end point selection.
    """
    # Compute the distance map from the target subvolume
    distance_map_gpu = cp_ndimage.distance_transform_edt(1 - target_subvolume_gpu)

    # Normalize the intensity image to match the scale of the distance map
    intensity_image_gpu = cp.asarray(intensity_image_gpu)
    # Create a mask for the object (1 where there's no object, 0 where there is)
    object_mask_gpu = 1 - object_subvolume_gpu

    # Apply the object mask to the intensity image
    intensity_image_gpu = intensity_image_gpu * object_mask_gpu

    normalized_intensity_gpu = intensity_image_gpu / cp.max(intensity_image_gpu)

    # Apply gamma correction to increase the influence of brighter voxels
    gamma = 0.4  # Adjust this value to fine-tune the brightness influence
    gamma_corrected_intensity = cp.power(normalized_intensity_gpu, gamma)

    # Create blurred intensity image
    blurred_intensity_gpu = cp_ndimage.gaussian_filter(gamma_corrected_intensity, sigma=[0.25, 10, 10])

    # Normalize the distance map to [0, 1]
    max_distance = cp.max(distance_map_gpu)
    normalized_distance_map_gpu = distance_map_gpu / (max_distance + 1e-6)

    # Invert the intensity so that brighter voxels have lower values (to minimize)
    intensity_weight = 0.2  # Adjust this weight to balance distance and intensity
    blurred_intensity_weight = 0.3
    epsilon = 1e-6
    augmented_map_gpu = normalized_distance_map_gpu / (
            intensity_weight * gamma_corrected_intensity +
            blurred_intensity_weight * blurred_intensity_gpu +
            epsilon
    )

    #augmented_map_gpu = (distance_map_gpu
    #                     - intensity_weight * gamma_corrected_intensity
    #                     - blurred_intensity_weight * blurred_intensity_gpu)



    # Apply Gaussian smoothing to reduce noise and create a smoother path
    sigma = [0.5, 0.5, 0.5]  # Adjust these values for each dimension
    augmented_map_gpu = cp_ndimage.gaussian_filter(augmented_map_gpu, sigma)

    # Convert augmented map, object, and target to NumPy for processing
    augmented_map = augmented_map_gpu.get()
    object_volume = object_subvolume_gpu.get()
    target_volume = target_subvolume_gpu.get()

    # Find potential start points (near the object)
    dilated_object = ndimage.binary_dilation(object_volume, iterations=1)
    start_candidates = np.argwhere(dilated_object & ~object_volume)

    # Find potential end points (on the target)

    modified_target = np.logical_xor(( ndimage.binary_dilation(target_volume, iterations=1)), target_volume)
    end_candidates = np.argwhere(modified_target)

    # Find the best start and end points
    best_start = min(start_candidates, key=lambda p: augmented_map[tuple(p)])
    best_end = min(end_candidates, key=lambda p: augmented_map[tuple(p)])

    # Initialize the path
    path = []
    current_point = tuple(best_start)
    path.append(current_point)

    max_iterations = 200  # Increased to allow for longer paths
    reached_target = False
    for _ in range(max_iterations):
        z, y, x = current_point

        # Get possible moves
        neighbors = [
            (int(z + dz), int(y + dy), int(x + dx))
            for dz in [-1, 0, 1] for dy in [-1, 0, 1] for dx in [-1, 0, 1]
            if (0 <= z + dz < augmented_map.shape[0] and
                0 <= y + dy < augmented_map.shape[1] and
                0 <= x + dx < augmented_map.shape[2] and
                not target_volume[z + dz, y + dy, x + dx])
        ]

        # Sort neighbors by the augmented map value
        neighbors.sort(key=lambda p: augmented_map[p])

        # Find the best move
        best_move = neighbors[0] if neighbors else None

        if best_move is None:
            # No valid moves available, terminate the path
            break

        z_only_move = (int(z + np.sign(best_move[0] - z)), y, x)

        # If moving only in Z is better or the same, prefer that
        if augmented_map[z_only_move] <= augmented_map[best_move] and not target_volume[z_only_move]:
            next_point = z_only_move
        else:
            next_point = best_move


            # Stop if the next point is at the modified target
        if modified_target[next_point]:
            path.append(next_point)
            reached_target = True
            break

        # Update current point and path
        current_point = next_point
        path.append(current_point)

    if reached_target:
        return path, augmented_map_gpu.get()
    else:
        #logger.warning("Path finding failed: did not reach the target.")
        return None, augmented_map_gpu.get()

def extend_objects_GPU(objects, target_objects, intensity, settings, locations, logger):
    logger.info("     Finding spine necks using GPU...")

    if not (objects.shape == target_objects.shape):
        raise ValueError("Input array shapes do not match")

    cp_objects = cp.asarray(objects)
    cp_necks = cp.zeros_like(objects)
    unique_labels = cp.unique(cp_objects)
    unique_labels = unique_labels[unique_labels != 0]

    full_shape = objects.shape
    for label_value in unique_labels:
        label_value = int(label_value.item())
        object_mask = cp_objects == label_value

        bbox = get_bounding_box_cupy(object_mask, 6, full_shape, settings)
        #bbox = get_bounding_box_with_target_cupy(object_mask, target_objects, full_shape, settings)
        bbox = tuple(map(int, bbox))

        object_sub_volume = cp.asnumpy(object_mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
        target_subvolume = cp.asnumpy(cp.asarray(target_objects)[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
        intensity_subvolume = cp.asnumpy(cp.asarray(intensity)[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])

        try:
            path_volume = 0
            result_label, path_volume, distance_vol = extend_single_object_GPU_v2(
                label_value, object_sub_volume, target_subvolume, intensity_subvolume, settings, logger)



            #print the max val in path volume
            #logger.info(f"Max value in path volume: {path_volume.max()}")

            cp_necks[bbox[0]:bbox[1],
            bbox[2]:bbox[3],
            bbox[4]:bbox[5]] = cp.maximum(
                cp_necks[bbox[0]:bbox[1],
                bbox[2]:bbox[3],
                bbox[4]:bbox[5]],
                cp.asarray(path_volume)
            )

            #logger.info(f"Extended object with label {result_label}. Bounding box: {bbox}")


            #logger.info(f"Subvolume shape: {object_sub_volume.shape}")
            #logger.info(f"Extended subvolume shape: {path_volume.shape}")
            #logger.info(f"Target subvolume shape: {target_subvolume.shape}")
            #logger.info(f"Traversable subvolume shape: {traversable_subvolume.shape}")
            save_neck_val_tifs = False
            if save_neck_val_tifs:
                tiff_filename = os.path.join(locations.Vols, f"subvol_{result_label}.tif")
                multichannel_subvolume = np.stack([
                    object_sub_volume,
                    distance_vol,
                    path_volume,
                    intensity_subvolume,
                    target_subvolume
                ], axis=1)
                imwrite(tiff_filename, multichannel_subvolume.astype(np.uint16), compression=('zlib', 1), imagej=True,
                        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX'})


        except Exception as e:
            logger.error(f"Error processing object with label {label_value}: {str(e)}")
            # Clean up memory for each loop iteration
            del object_mask, object_sub_volume, target_subvolume, intensity_subvolume, path_volume, distance_vol
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()

            # Clean up memory for variables that are no longer needed
    del cp_objects, unique_labels
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(cp_necks)


def extend_single_object_GPU_v2(label_value, object_subvolume, target_subvolume, intensity_subvolume, settings, logger):
    # Convert numpy arrays to CuPy arrays
    # logger.info(f"Label {label_value} - Subvolume shapes: subvolume {object_subvolume.shape}, target {target_subvolume.shape}, traversable {traversable_subvolume.shape}")

    pad_width = 1

    object_subvolume_gpu = pad_subvolume_gpu(cp.asarray(object_subvolume), pad_width)
    target_subvolume_gpu = pad_subvolume_gpu(cp.asarray(target_subvolume), pad_width)
    # traversable_subvolume_gpu = cp.asarray(traversable_subvolume)
    intensity_subvolume_gpu = pad_subvolume_gpu(cp.asarray(intensity_subvolume), pad_width)
    # Use the enhanced simple pathfinding method - below method works well but seeing if we can use intensity as well
    # path, distance_map = strict_z_first_pathfinding(object_subvolume_gpu, target_subvolume_gpu)

    path, distance_map = pathfinding_v3b(object_subvolume_gpu, target_subvolume_gpu, intensity_subvolume_gpu,
                                               logger)

    #imwrite distance map
    #ast ype float
    #distance_map = distance_map.astype(np.float32)
    #imwrite(f"D:/Project_Data/RESPAN/Testing/_2024_08_Test_with_Spines/1/Validation_Data/distance_map{label_value}.tif", distance_map.astype(np.float32), imagej=True, photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})
    # if not path:
    #    logger.info(f"No path found for label {label_value}.")
    # else:
    #    logger.info(f"Path length: {len(path)}")
    '''
    #logger.info(f"Start point: {start_point}, End point: {end_point}")
    #logger.info(f"Cost array slice: {cost_array[start_point[0], start_point[1], start_point[2]]}")
    #logger.info(f"Path found: {path}")
    # Create extended subvolume

    path_volume = cp.copy(object_subvolume_gpu)
    if path:
        path_array = np.array(path)
        logger.info(f"Updating path_volume at indices: {path_array.T}")
        path_volume[tuple(path_array.T)] = label_value
    #print label value
    logger.info(f"Label value: {label_value}")

    path_volume = cp.logical_xor(object_subvolume_gpu, path_volume)
    path_volume = path_volume * label_value
    '''
    # Create the path volume
    path_volume_gpu = cp.zeros_like(object_subvolume_gpu)
    if path is not None:
        path_array = np.array(path, dtype=int)  # Ensure path array is of integer type
        # logger.info(f"Updating path_volume at indices: {path_array.T}")

        # Limit the extension to 10 voxels - added to prevent long paths
        max_distance = 20
        path_array = path_array[distance_map[tuple(path_array.T)] <= max_distance]
        path_volume_gpu[tuple(path_array.T)] = 1
    else:
        return label_value, cp.zeros_like(object_subvolume), cp.zeros_like(object_subvolume)

    # Subtract the path from the object subvolume using logical XOR
    final_path_volume_gpu = path_volume_gpu * (1 - object_subvolume_gpu)

    # Multiply the final path volume by the label value
    final_path_volume_gpu = final_path_volume_gpu * label_value

    # Convert to NumPy after the operation
    path_volume = unpad_subvolume_gpu(cp.asnumpy(final_path_volume_gpu), pad_width)
    distance_map = unpad_subvolume_gpu(cp.asnumpy(distance_map), pad_width)


    return label_value, path_volume, distance_map

def pad_subvolume_gpu(subvolume_gpu, pad_width):
    subvolume_gpu = cp.asarray(subvolume_gpu)
    return cp.pad(subvolume_gpu, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)),
                  mode='constant', constant_values=0)

def unpad_subvolume_gpu(subvolume_gpu, pad_width):
    return subvolume_gpu[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

def spine_neck_analysis_gpu_batch(subvolumes, logger, scaling):

    cp_subvolumes = [cp.asarray(sv) for sv in subvolumes]

    results = []

    for cp_subvolume in cp_subvolumes:

        # Extract surface using marching cubes
        verts, faces, _, _ = marching_cubes(cp.asnumpy(cp_subvolume))

        # Convert vertices and faces to CuPy arrays
        verts = cp.asarray(verts) * cp.asarray(scaling)
        faces = cp.asarray(faces)

        # Compute volume using triangles
        volume = mesh_volume(verts, faces)

        # compute length and then widths
        length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length(cp_subvolume, verts, logger, scaling)
        # Compute length using the longest path
        #length = mesh_length(verts, faces)

        # Compute width statistics
        #min_width, max_width, mean_width = mesh_neck_width(verts, faces)
        results.append((volume, length, min_width, max_width, mean_width))

    return results #volume, length, min_width, max_width, mean_width


def mesh_volume(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    volume = cp.sum(cp.cross(v0, v1) * v2) / 6.0
    return abs(float(volume))


def smooth_skeleton_3d(skeleton_points, num_samples=50):
    """
    Smooth the 3D skeleton using spline interpolation.
    """
    if len(skeleton_points) < 4:
        return cp.asarray(skeleton_points)  # Not enough points for smoothing

    try:
        # Use NumPy for spline fitting as SciPy doesn't support CuPy arrays
        #tck, u = splprep([skeleton_points[:, 0].get(), skeleton_points[:, 1].get(), skeleton_points[:, 2].get()], s=0, k=3)
        tck, u = splprep([skeleton_points[:, i].get() for i in range(3)], s=0, k=3)
        u_fine = np.linspace(0, 1, num_samples)
        smoothed = np.array(splev(u_fine, tck)).T
        return cp.asarray(smoothed)
    except Exception as e:
        print(f"Spline fitting failed: {e}. Using original points.")
        return skeleton_points


def calculate_neck_length_and_width(skeleton_points, verts):
    """
    Calculate neck length and width relative to the skeleton.

    """
    # Smooth the 3D skeleton
    smooth_skeleton_points = smooth_skeleton_3d(skeleton_points)

    # Calculate the total length of the neck
    diff = cp.diff(smooth_skeleton_points, axis=0)
    total_length = float(cp.sum(cp.sqrt(cp.sum(diff ** 2, axis=1))))

    # Calculate tangent vectors along the skeleton
    tangents = cp.diff(smooth_skeleton_points, axis=0)
    tangents = tangents / cp.linalg.norm(tangents, axis=1)[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))  # Add last tangent

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane

    widths = cp.zeros(len(smooth_skeleton_points))
    for i, (skeleton_point, tangent) in enumerate(zip(smooth_skeleton_points, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    # Compute statistics
    min_width = float(cp.min(widths))
    max_width = float(cp.max(widths))
    mean_width = float(cp.mean(widths))

    return total_length, min_width, max_width, mean_width, smooth_skeleton_points


def mesh_neck_width_and_length(volume, verts, logger, scaling):
    """
    Calculate the neck length and width using the skeleton and vertices.

    Args:
        volume (ndarray): 3D binary volume representing the object.
        verts (ndarray): Vertices representing the surface or points of the object.
        scaling (tuple): Voxel scaling in (Z, Y, X) order.

    Returns:
        total_length (float): Total length of the neck.
        min_width (float): Minimum width of the neck.
        max_width (float): Maximum width of the neck.
        mean_width (float): Mean width of the neck.
    """
    #print logger and scaling
    #logger.info(f"Calculating neck length and width with scaling: {scaling}")
    skeleton = morphology.skeletonize(cp.asnumpy(volume))
    skeleton_points = cp.asarray(np.argwhere(skeleton)) * cp.asarray(scaling)
    #skeleton_points = np.argwhere(skeleton)

    if len(skeleton_points) == 0:
        # No skeleton points, return 0 for all metrics
        return 0, 0, 0, 0, 0
        # Apply voxel scaling to the vertices
    #skeleton_points_scaled = skeleton_points * cp.asarray(scaling)

    if len(skeleton_points) < 2:

        # Use two skeleton points to form a basic axis for width measurement
        # Treat as a straight line between the two bounding box corners
        bbox_min = np.min(verts, axis=0)
        bbox_max = np.max(verts, axis=0)

        # Skeleton axis: straight line between bounding box corners
        skeleton_axis = bbox_max - bbox_min
        skeleton_axis /= np.linalg.norm(skeleton_axis)  # Normalize axis

        # Project the vertices onto the skeleton axis to measure distances perpendicular to the axis
        diffs = verts - bbox_min  # Differences relative to one endpoint of the axis
        projections = cp.dot(diffs, skeleton_axis)[:, None] * skeleton_axis  # Project onto axis
        perpendicular_diffs = verts - (bbox_min + projections)  # Perpendicular vectors

        # Compute the width in the XY plane (ignore Z, only use Y and X for width)
        perpendicular_diffs_xy = perpendicular_diffs[:, 1:]  # Only Y and X
        dists = cp.linalg.norm(perpendicular_diffs_xy, axis=1)

        # Compute width statistics (min, max, mean) in the XY plane
        min_width = float(cp.min(dists))
        max_width = float(cp.max(dists))
        mean_width = float(cp.mean(dists))

        # Compute the total length of the neck as the bounding box diagonal
        total_length = np.linalg.norm(bbox_max - bbox_min)
        #total_length *= np.linalg.norm(scaling)  # Adjust for voxel scaling
        #create a line based on bbox min and max that can be plotted
        #bbox_min are cp we need np

        skeleton_output = np.array([bbox_min.get(), bbox_max.get()])



    else:
        total_length, min_width, max_width, mean_width, skeleton_output = calculate_neck_length_and_width(
            skeleton_points, verts)

    # Calculate neck length and width using the smooth skeleton
    return total_length, min_width, max_width, mean_width, skeleton_output



def mesh_neck_width_and_length_midline_calculated(midline, verts, logger, scaling):

    if len(midline) == 1:

        #print("Midline is too short to calculate tangents")
        return scaling[1], scaling[1], scaling[1], scaling[1], cp.array(midline)

    scaling = cp.array(scaling)
    # Calculate total length
    total_length = cp.sum(cp.sqrt(cp.sum(cp.diff(midline * scaling, axis=0) ** 2, axis=1)))
    #print(f"Total length: {total_length}")

    # Calculate tangent vectors along the midline
    tangents = cp.diff(midline, axis=0)
    tangent_norms = cp.linalg.norm(tangents, axis=1)
    tangent_norms = cp.where(tangent_norms == 0, 1e-8, tangent_norms)
    tangents = tangents / tangent_norms[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))
    #print(f"Tangents shape: {tangents.shape}")

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane
    widths = cp.zeros(len(midline))
    for i, (skeleton_point, tangent) in enumerate(zip(midline, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    #print(f"Widths shape: {widths.shape}")
    if len(widths) == 0:
       #print("No widths calculated")
        return total_length, 0, 0, 0, None

    # Compute statistics
    min_width = float(cp.min(widths)) * scaling[1]
    max_width = float(cp.max(widths)) * scaling[1]
    mean_width = float(cp.mean(widths)) * scaling[1]


    # Calculate neck length and width using the smooth skeleton
    #print(midline)
    return total_length, min_width, max_width, mean_width, midline

def mesh_neck_width_and_length_closest(volume, verts, closest_point, logger, scaling):
    """
    Calculate the neck length and width using the skeleton and vertices.

    """
    #print(f"Input volume shape: {volume.shape}")
    #print(f"Number of vertices: {len(verts)}")
    #print(f"Closest point: {closest_point}")
    #print(f"Scaling: {scaling}")

    if cp.sum(volume) == 0:
        return 0, 0, 0, 0, None
    elif cp.sum(volume) == 1:
        #print("Volume contains only one voxel")
        return 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], cp.argwhere(volume)

    if cp.isnan(volume).any() or cp.isinf(volume).any():
        logger.error("Volume contains NaN or Inf values")
        return 0, 0, 0, 0, None

    # Create distance map from the object surface
    mask = cp.asnumpy(volume>0)
    dist_map = distance_transform_edt(mask, sampling=scaling)
    dist_map = cp.asarray(dist_map)
    #print(f"Distance map shape: {dist_map.shape}")
    #print max value in dist_map
    #print(f"Max value in dist_map: {cp.max(dist_map)}")
    if cp.isnan(dist_map).any() or cp.isinf(dist_map).any():
        logger.error("Distance map contains NaN or Inf values")
        return 0, 0, 0, 0, None

    # Find the furthest point from the closest_point
    closest_point = tuple(map(int, closest_point))
    #print(f"Closest point: {closest_point}")

    temp_map = dist_map.copy()
    temp_map[~mask] = 0
    furthest_point = cp.unravel_index(cp.argmax(temp_map), volume.shape)
    #print(f"Furthest point: {furthest_point}")

    # Create a cost map (inverse of distance map)
    cost_map = cp.max(dist_map) - dist_map
    cost_map[volume == 0] = cp.inf  # Set cost to infinity outside the object

    # More aggressive boosting around the furthest point
    gaussian_boost_radius = 3  # Smaller sigma for sharper boost
    boost_map = cp.zeros_like(cost_map)

    # Set the furthest point in the boost_map as 1 (center of the Gaussian)
    boost_map[furthest_point] = 1.0

    # Apply Gaussian filter to smooth the boost map, creating a sharper boost
    boost_map = cp.asarray(gaussian_filter(cp.asnumpy(boost_map), sigma=gaussian_boost_radius))

    # Now modify the cost map with a stronger Gaussian boost
    cost_map = cost_map * (1.0 - boost_map * 2) + boost_map * 0.1  # More aggressive scaling near furthest point

    # Convert to NumPy for route_through_array
    cost_map_np = cp.asnumpy(cost_map)
    furthest_point_np = tuple(int(i) for i in furthest_point)
    #print(f"Furthest point (NP): {furthest_point_np}")

    # Find the optimal path through the center of the object
    try:
        indices, _ = route_through_array(cost_map_np, closest_point, furthest_point_np, fully_connected=True)
        if not indices:
            raise ValueError("No path found between closest and furthest points")
    except Exception as e:
        #print(f"Error in route_through_array: {e}")
        return 0, 0, 0, 0, None

    midline = cp.array(indices)
    #midline = extend_midline_to_boundary(midline, volume)
    #midline = extend_midline_to_furthest_point(midline, furthest_point, volume, logger)


    # Geodesic distance map from the furthest point
    #binary_volume = volume > 0
    #geodesic_map = distance_transform_edt(cp.asnumpy(binary_volume), sampling=scaling)  # Use NumPy for geodesic map


    # Follow the geodesic path from furthest_point to the midline
    #midline = extend_to_furthest_point_geodesic(midline, geodesic_map, furthest_point)


    if len(midline) < 2:
        #print("Midline is too short to calculate tangents")
        return 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], 1 * scaling[1], cp.argwhere(volume)

    scaling = cp.array(scaling)
    # Calculate total length
   # total_length = cp.sum(cp.sqrt(cp.sum(cp.diff(midline * scaling, axis=0) ** 2, axis=1)))
    # Calculate the differences between consecutive midline points
    diff = cp.diff(midline, axis=0)

    # Apply scaling to each axis independently to account for anisotropy
    scaled_diff = diff * scaling  # Element-wise multiplication of voxel differences by the corresponding scaling factors

    # Calculate the anisotropic length by summing the Euclidean distance with correct scaling for each axis
    distances = cp.sqrt(cp.sum(scaled_diff ** 2, axis=1))

    # Sum the distances to get the total length
    total_length = cp.sum(distances)
    #print(f"Total length: {total_length}")

    # Calculate tangent vectors along the midline
    tangents = cp.diff(midline, axis=0)
    tangent_norms = cp.linalg.norm(tangents, axis=1)
    tangent_norms = cp.where(tangent_norms == 0, 1e-8, tangent_norms)
    tangents = tangents / tangent_norms[:, None]
    tangents = cp.vstack((tangents, tangents[-1]))
    #print(f"Tangents shape: {tangents.shape}")

    # Function to compute perpendicular distance in XY plane
    @cp.fuse()
    def perpendicular_distance_xy(point_y, point_x, skeleton_y, skeleton_x, tangent_y, tangent_x):
        vec_y = point_y - skeleton_y
        vec_x = point_x - skeleton_x
        proj = vec_y * tangent_y + vec_x * tangent_x
        perp_vec_y = vec_y - proj * tangent_y
        perp_vec_x = vec_x - proj * tangent_x
        return cp.sqrt(perp_vec_y ** 2 + perp_vec_x ** 2)

    # Calculate widths in XY plane
    widths = cp.zeros(len(midline))
    for i, (skeleton_point, tangent) in enumerate(zip(midline, tangents)):
        distances = perpendicular_distance_xy(
            verts[:, 1], verts[:, 2],
            skeleton_point[1], skeleton_point[2],
            tangent[1], tangent[2]
        )
        widths[i] = cp.max(distances) * 2  # Diameter

    #print(f"Widths shape: {widths.shape}")
    if len(widths) == 0:
       #print("No widths calculated")
        return total_length, 0, 0, 0, None

    # Compute statistics
    min_width = float(cp.min(widths)) * scaling[1]
    max_width = float(cp.max(widths)) * scaling[1]
    mean_width = float(cp.mean(widths)) * scaling[1]


    # Calculate neck length and width using the smooth skeleton
    #print(midline)
    return total_length, min_width, max_width, mean_width, midline



def extract_subvolumes_GPU_batch(labeled_array, labels, padding=5):
    """
    Extract subvolumes for multiple labels in parallel on GPU using CuPy.

    Args:
        labeled_array (cp.ndarray): Labeled array where each unique label represents a distinct region.
        labels (cp.ndarray): Array of unique labels to extract.
        padding (int): Padding to apply around the bounding box of each labeled region.

    Returns:
        subvolumes (list of cp.ndarray): List of subvolumes for each label.
        coords (list of cp.ndarray): List of start coordinates for each subvolume.
    """
    subvolumes = []
    start_coords = []

    # Convert to CuPy array if it's not already
    labeled_array = cp.asarray(labeled_array)
    labels = cp.asarray(labels)

    # Get binary masks for all labels in one batch operation
    binary_masks = labeled_array[:, :, :, None] == labels[None, None, None, :]  # Shape: (z, y, x, n_labels)

    # Iterate through the labels and process each label's subvolume in parallel
    for idx, label in enumerate(labels):
        # Find the coordinates of the current label
        #print(f"Processing label {label}...")
        binary_mask = binary_masks[..., idx]
        coords = cp.argwhere(binary_mask)

        # Check if there are any coordinates for the current label
        if coords.size == 0:
            continue

        # Compute the start and end coordinates with padding
        start = cp.maximum(cp.min(coords, axis=0) - padding, 0)
        end = cp.minimum(cp.max(coords, axis=0) + padding + 1, cp.array(labeled_array.shape))

        # Extract the subvolume for the current label
        subvolume = labeled_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]] == label

        # Store the subvolume and the start coordinates
        subvolumes.append(subvolume)
        start_coords.append(start)

    return subvolumes, cp.stack(start_coords)

def extract_subvolumes_mulitchannel_GPU_batch(multi_channel_array, labels, padding=5):
    subvolumes = []
    start_coords = []

    # Get binary masks for all labels in one batch operation
    binary_masks = multi_channel_array[:, :, :, 0, None] == labels[None, None, None, :]

    for idx, label in enumerate(labels):
        binary_mask = binary_masks[..., idx]
        coords = cp.argwhere(binary_mask)

        if coords.size == 0:
            continue

        start = cp.maximum(cp.min(coords, axis=0) - padding, 0)
        end = cp.minimum(cp.max(coords, axis=0) + padding + 1, cp.array(multi_channel_array.shape[:3]))

        subvolume = multi_channel_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]



        #mask the subvolume
        #label_mask = (subvolume[:, :, :, 0] == label)
        #label_mask2 = (subvolume[:, :, :, 1] == label)

        # Apply the mask to channels 1 and 2, converting them to binary
        #subvolume[:, :, :, 0] = subvolume[:, :, :, 0] == label # Channel 1 becomes binary mask
        #subvolume[:, :, :, 1] = subvolume[:, :, :, 1] == label  # Channel 2 becomes binary mask
        #just mask keep label
        #subvolume[:, :, :, 0] = cp.where(label_mask, subvolume[:, :, :, 0], 0)
        #subvolume[:, :, :, 1] = cp.where(label_mask, subvolume[:, :, :, 1], 0)

        subvolumes.append(subvolume)
        start_coords.append(start)

    return subvolumes, cp.stack(start_coords)


def extract_subvolumes_mulitchannel_GPU_batch_2ndpass(multi_channel_array, labels, padding=5):
    subvolumes = []
    start_coords = []

    # Get binary masks for all labels in one batch operation
    binary_masks = multi_channel_array[:, :, :, 0, None] == labels[None, None, None, :]

    for idx, label in enumerate(labels):
        binary_mask = binary_masks[..., idx]
        coords = cp.argwhere(binary_mask)

        if coords.size == 0:
            continue

        start = cp.maximum(cp.min(coords, axis=0) - padding, 0)
        end = cp.minimum(cp.max(coords, axis=0) + padding + 1, cp.array(multi_channel_array.shape[:3]))

        subvolume = multi_channel_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]



        #mask the subvolume
        #label_mask = (subvolume[:, :, :, 0] == label)
        #label_mask2 = (subvolume[:, :, :, 1] == label)

        # Apply the mask to channels 1 and 2, converting them to binary
        #subvolume[:, :, :, 0] = subvolume[:, :, :, 0] == label # Channel 1 becomes binary mask
        #subvolume[:, :, :, 1] = subvolume[:, :, :, 1] == label  # Channel 2 becomes binary mask
        #just mask keep label
        #subvolume[:, :, :, 0] = cp.where(label_mask, subvolume[:, :, :, 0], 0)
        #subvolume[:, :, :, 1] = cp.where(label_mask, subvolume[:, :, :, 1], 0)

        subvolumes.append(subvolume)
        start_coords.append(start)

    return subvolumes, cp.stack(start_coords)



def import_tiff_files_to_cupy_list(folder_path):
    """
    Import all TIFF files from a folder into a list of CuPy arrays.

    Returns:
    list: A list of CuPy arrays, each representing a TIFF file.
    """
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
    tiff_files.sort()  # Ensure consistent ordering

    imported_list = []

    for filename in tiff_files:
        file_path = os.path.join(folder_path, filename)
        # Read the TIFF file and convert to CuPy array
        image = cp.asarray(imread(file_path))


        imported_list.append(image)

    return imported_list


def second_pass_annotation(head_labels_vol, neck_labels_vol, dendrite_vol, neuron_vol, locations, settings, logger):
    #dendrites and labels and must be binarized for further analysis
    #ensure data is rescaled to match second pass model

    start_time = time.time()
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    logger.info(f"       Available GPU memory: {free_mem / 1e9:.2f} GB")
    # estimate requirements
    avg_size = 8 / settings.refinement_Z * 8 /  settings.refinement_XY * 8 /  settings.refinement_XY  # spine or neck
    batch_size = max(1, int(free_mem * 0.7 / (avg_size * 4)))  # 4 bytes per float32
    #print(f"Using batch size: {batch_size}")

    # Move data to GPU and create multi-channel array
    cp_multi_channel = cp.stack([cp.asarray(head_labels_vol+neck_labels_vol),
                                            cp.asarray(neck_labels_vol > 0),
                                            cp.asarray(dendrite_vol >0),
                                            cp.asarray(neuron_vol)], axis=-1)

    # Get unique labels from array A (excluding background)
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]



      #batch
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        logger.info(f"      Processing batch {i // batch_size + 1} of {len(labels) // batch_size + 1}...")

        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch_2ndpass(cp_multi_channel, batch_labels)

        #save each subvolume as a tiff in a folder
        for subvol, label in zip(sub_volumes, batch_labels):


            # save as tif and resahpe for imageJ
            imwrite_filename = os.path.join(locations.nnUnet_2nd_pass, f"subvol_{10000+label}_0000.tif")
            # imageJ ZCYX - currently ZYXC so fix

            # subvol_out = subvol.get()
            imwrite(imwrite_filename, subvol[:,:,:,3].get().transpose(0, 1, 2).astype(np.uint16), compression=('zlib', 1), imagej=True,
                    photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZYX'})

            #process with nnunet

        # split the path into subdirectories
        subdirectories = os.path.normpath(settings.refinement_model_path).split(os.sep)
        last_subdirectory = subdirectories[-1]
        # find all three digit sequences in the last subdirectory
        matches = re.findall(r'\d{3}', last_subdirectory)
        # If there's a match, assign it to a variable
        dataset_id = matches[0] if matches else None

        logger.info("\nPerforming spine refinement on GPU...\n")
        #create dir locations.nnUnet_2nd_pass+'\labels'
        if not os.path.exists(locations.nnUnet_2nd_pass+'\labels'):
            os.makedirs(locations.nnUnet_2nd_pass+'\labels')

        ##uncomment if issues with nnUnet
        # logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")

        return_code = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env,
                                         locations.nnUnet_2nd_pass, locations.nnUnet_2nd_pass+'\labels', dataset_id, settings.nnUnet_type,
                                         settings, logger)

        logger.info(f"Updating subvolumes with refined labels...")

        #import all tifs in output folder as updated_sub_volumes .tif
        updated_sub_volumes = import_tiff_files_to_cupy_list(locations.nnUnet_2nd_pass+'\labels')

        multi_channel_subvolumes = []

        for subvolume, label in zip(updated_sub_volumes, labels):
            # Create boolean masks for each channel
            c0 = (subvolume == 1).astype(cp.float32) * label  # Multiply by label
            c1 = (subvolume == 2).astype(cp.float32)
            c2 = (subvolume == 4).astype(cp.float32)  * label  # Multiply by label

            # Stack the channels to create a multi-channel array
            multi_channel = cp.stack([c0, c1, c2], axis=-1)

            multi_channel_subvolumes.append(multi_channel)
        # Clean up label folder
        # delete nnunet input folder and files
        if settings.save_intermediate_data == False:

            shutil.rmtree(locations.nnUnet_2nd_pass)
            shutil.rmtree(locations.nnUnet_2nd_pass+'\labels')

        # Insert processed subvolumes back into the original image
        logger.info(f"Inserting refined subvolumes back into the original image...")
        cp_multi_channel = insert_subvolumes(cp_multi_channel, multi_channel_subvolumes, start_coords, labels)

    return cp_multi_channel


def insert_subvolumes(original_image, processed_subvolumes, start_coords, labels):
    # Create a mask for the areas we've modified
    modified_mask = cp.zeros(original_image.shape[:3], dtype=bool)

    for subvolume, start, label in zip(processed_subvolumes, start_coords, labels):
        end = start + cp.array(subvolume.shape[:3])

        # Create a mask for the current label in the original image
        label_mask = original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] == label

        # Update the modified mask
        modified_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] |= label_mask

        # Clear the areas corresponding to the current label in c0 and c1
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] = cp.where(label_mask, 0,
                                                                                        original_image[start[0]:end[0],
                                                                                        start[1]:end[1],
                                                                                        start[2]:end[2], 0])
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 1] = cp.where(label_mask, 0,
                                                                                        original_image[start[0]:end[0],
                                                                                        start[1]:end[1],
                                                                                        start[2]:end[2], 1])

        # Add the subvolume data where it's non-zero
        subvolume_mask = subvolume > 0
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 0] += cp.where(subvolume_mask[..., 0],
                                                                                         subvolume[..., 0], 0)
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 1] += cp.where(subvolume_mask[..., 2],
                                                                                         subvolume[..., 2], 0)

        # For c2, combine using logical OR
        original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 2] = cp.logical_or(
            original_image[start[0]:end[0], start[1]:end[1], start[2]:end[2], 2],
            subvolume[..., 1] > 0
        ).astype(original_image.dtype)

    # Clear any remaining voxels in c0 and c1 that weren't replaced
    original_image[:, :, :, 0] = cp.where(modified_mask, original_image[:, :, :, 0], 0)
    original_image[:, :, :, 1] = cp.where(modified_mask, original_image[:, :, :, 1], 0)

    return original_image


def log_memory_usage(logger):
    # Get current process
    process = psutil.Process(os.getpid())

    # Log overall memory usage
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MiB
    logger.info(f"Current memory usage: {mem_usage:.6f} MiB")

    # Log detailed memory usage
    logger.info("Detailed memory usage:")
    mem_usage = memory_profiler.memory_usage()
    logger.info(f"Total memory usage: {mem_usage[0]:.6f} MiB")

def check_gpu_memory_requirements(spine_labels_vol, logger):
    # Calculate memory for one volume
    single_vol_memory = spine_labels_vol.nbytes

    # Calculate memory for all four volumes
    total_memory_required = single_vol_memory * 4

    # Get total GPU memory
    free_mem, total_gpu_memory = cp.cuda.runtime.memGetInfo()

    #logger.info(f"       Memory required for one volume: {single_vol_memory / 1e9:.2f} GB")
    logger.info(f"       Total memory required for full multi-channel array: {total_memory_required / 1e9:.2f} GB")
    logger.info(f"       Total available GPU memory: {free_mem / 1e9:.2f} GB. Percentage of GPU memory required for array: {(total_memory_required / free_mem) * 100:.2f}%")

    return total_memory_required, free_mem, total_gpu_memory

def calculate_batch_size(mem_required, labels, sub_volume_shape, dtype_size):
    # Get the default memory pool
    memory_pool = cp.get_default_memory_pool()
    # Get total and used bytes from the memory pool
    used_bytes = memory_pool.used_bytes()
    total_bytes = cp.cuda.Device().mem_info[1]

    # Calculate free memory in bytes
    free_mem = total_bytes - used_bytes

    # Calculate memory required for one sub-volume (4 channels)
    sub_volume_memory = np.prod(sub_volume_shape) * dtype_size * 4  # bytes

    # Estimate additional overhead (temporary arrays, etc.)
    # Adjust the overhead factor based on empirical observations
    overhead_factor = 3
    memory_per_item = sub_volume_memory * overhead_factor

    # Calculate batch size, leaving 20% of free memory as buffer
    available_memory = (free_mem - mem_required) * 0.8
    batch_size = int(available_memory / memory_per_item)
    batch_size = max(1, min(batch_size, len(labels)))

    return batch_size

def calculate_batch_size_v0(mem_required, labels, sub_volume_shape, dtype_size):
    free_mem, _ = cp.cuda.runtime.memGetInfo()

    # Calculate memory required for one sub-volume
    sub_volume_memory = np.prod(sub_volume_shape) * dtype_size *4  # 4 channels

    # Calculate memory required for other operations
    other_operations_memory = sub_volume_memory * 2

    # Total memory per item in batch
    memory_per_item = sub_volume_memory + other_operations_memory

    # Calculate batch size, leaving 20% of free memory as buffer
    batch_size = int(((free_mem-mem_required) * 0.8) / memory_per_item)

    return int(max(1, min(batch_size, len(labels)))/2.5)

def calculate_tile_size(shape, scaling, logger):
    _, total_mem = cp.cuda.runtime.memGetInfo()
    target_mem = 0.25 * total_mem  # 25% of total GPU memory
    voxel_size = 4 * 4  # 4 volumes, 4 bytes per voxel (assuming float32)
    total_voxels = target_mem / voxel_size

    # Calculate z_size to be 1/50th of y and x
    ratio = 50
    z_size = int((total_voxels / (ratio * ratio)) ** (1/3))
    y_size = x_size = z_size * ratio

    # Adjust sizes to fit within the original shape
    z_size = min(z_size, shape[0])
    y_size = min(y_size, shape[1])
    x_size = min(x_size, shape[2])

    # Ensure sizes are at least 1
    z_size = max(1, z_size)
    y_size = max(1, y_size)
    x_size = max(1, x_size)

    logger.info(f"     Calculated tile size: z={z_size}, y={y_size}, x={x_size}")
    return (z_size, y_size, x_size)


def process_spine_batch_tile(tile, tile_spine, tile_head, tile_dendrite, tile_neuron, locations, settings, logger, scaling):
    # Create multi-channel array for the tile

    mem_required, free_mem, total_mem = check_gpu_memory_requirements(tile_spine, logger)

    cp_multi_channel = cp.stack([cp.asarray(tile_spine),
                                 cp.asarray(tile_head),
                                 cp.asarray(tile_dendrite),
                                 cp.asarray(tile_neuron)], axis=-1)

    # Get unique labels from the tile
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]

    logger.info(f"       Processing tile {tile} with {len(labels)} spines...")

    # Calculate batch size for this tile
    avg_sub_volume_shape = (6 / scaling[0], 6 / scaling[1], 6 / scaling[2])
    dtype_size = 2  # Assuming float32
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    batch_size = calculate_batch_size(mem_required, tile_spine, avg_sub_volume_shape, dtype_size)
    #logger.info(f"       Using batch size of {batch_size} spines... free_mem: {mem_required} avg_sub_volume_shape, dtype_size {avg_sub_volume_shape} {dtype_size}")

    tile_results = []

    if batch_size == 0:
        return  tile_results

    logger.info(f"       Using batch size of {batch_size} spines...")



    # batch
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        logger.info(
            f"        Processing batch {i // batch_size + 1} of {len(labels) // batch_size + 1}...")

        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch(cp_multi_channel, batch_labels)

        # logger shape
        if settings.additional_logging == True:
            logger.info(f"Subvolumes shape: {len(sub_volumes)}")
            # log dimensions of subvolumes0
            logger.info(f"Subvolume 0 shape: {sub_volumes[0].shape}")

        # mask subvolumes
        # logger.info(f"      Masking subvolumes...")
        sub_volumes = batch_mask_subvolumes_cp(sub_volumes, batch_labels)

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after masking: {sub_volumes[0].shape}")

        # Pad subvolumes to allow batching
        # padded_subvolumes, pad_amounts = pad_subvolumes(sub_volumes, logger)

        padded_subvolumes, pad_amounts = pad_and_center_subvolumes(sub_volumes, settings, logger)

        # Save spine arrays - shape is M
        if settings.additional_logging == True:
            logger.info(f"      Saving spine arrays...")
        # MZYXC to MZCYX for imagej
        export_subvols = cp.transpose(cp.stack(padded_subvolumes, axis=0), (0, 1, 4, 2, 3))
        export_subvols[:, :, :3] = export_subvols[:, :, :3] * 65535 / 2
        export_subvols[:, :, 0] = export_subvols[:, :, 0] - export_subvols[:, :, 1]
        export_subvols = cp.asnumpy(export_subvols)
        # logger.info(export_subvols.shape)

        imwrite(f'{locations.arrays}/Spine_vols_{settings.filename}_tile{tile}_b{i // batch_size + 1}.tif',
                export_subvols.astype(np.uint16), compression=('zlib', 1), imagej=True,
                photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX', 'mode': 'composite'},
                resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        imwrite(f'{locations.arrays}/Spine_MIPs_{settings.filename}_tile{tile}_b{i // batch_size + 1}.tif',
                np.max(export_subvols, axis=1).astype(np.uint16), compression=('zlib', 1), imagej=True,
                photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        # imwrite(f'{locations.arrays}/Spine_slices_{settings.filename}_b{i // batch_size + 1}.tif',
        #        export_subvols[:,export_subvols.shape[1] // 2].astype(np.uint16), imagej=True,
        #        photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
        #        resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        del export_subvols

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after padding: {padded_subvolumes[0].shape}")

        # loop over sub_volumes but also provide a counter
        # if settings.save_intermediate_data == True:
        #    for subvol, start_coord, label in zip(sub_volumes, start_coords, batch_labels):

        # save as tif and resahpe for imageJ
        #        imwrite_filename = os.path.join(locations.Meshes, f"subvol_{label}.tif")
        # imageJ ZCYX - currently ZYXC so fix

        # subvol_out = subvol.get()
        #        imwrite(imwrite_filename, subvol.get().transpose(0, 3, 1, 2).astype(np.uint16), imagej=True,
        #                photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZCYX'})

        # generate data for spine
        batch_data = cp.stack([subvol[:, :, :, 0].astype(cp.float32) for subvol in padded_subvolumes])  # spine volumes
        batch_dendrite = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes])  # dendrite
        # binary of batch_dendrite
        batch_binary = cp.stack([subvol[:, :, :, 2] > 0 for subvol in padded_subvolumes])  # dendrite binary

        # calculate closest points
        closest_points = find_closest_points_batch(batch_data, batch_binary)

        #dendrite_ids = find_closest_dendrite_id(closest_points, batch_dendrite)
        # print(dendrite_ids)
        if settings.additional_logging == True:
            logger.info(f"batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")
            logger.info(f"batch_dendrite dtype: {batch_dendrite.dtype}, shape: {batch_dendrite.shape}")
            logger.info(f"closest_points dtype: {closest_points.dtype}, shape: {closest_points.shape}")
            #logger.info(f"dendrite_ids dtype: {dendrite_ids.dtype}, shape: {dendrite_ids.shape}")

        # Process spines and pass midlines on for neck and head analysis
        t0 = time.time()

        spine_results, midlines = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts,
                                                      batch_labels, scaling, 0, True, None, True, True, "spine",
                                                      locations, settings, logger)
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spines: {time.time() - t0:.2f} seconds")

        # Subtract heads from spines and measure necks and ensure float for marching_cubes
        batch_data = cp.stack([
            cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0).astype(cp.float32)
            for subvol in padded_subvolumes
        ])

        # for subvol in sub_volumes:
        #    subvol[:, :, :, 0] = cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0)
        t0 = time.time()

        neck_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels,
                                              scaling, 0, False, midlines, True, True, "neck", locations, settings,
                                              logger)  # True for min max mean width
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of necks: {time.time() - t0:.2f} seconds")

        # now process heads
        t0 = time.time()
        batch_data = cp.stack([subvol[:, :, :, 1] for subvol in padded_subvolumes])  # spine vols
        # Process just heads - but don't need vol for length just surface area
        head_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels,
                                              scaling, 1, False, midlines, True, True, "head", locations, settings,
                                              logger)  # False for width using bounding box
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spine heads: {time.time() - t0:.2f} seconds")
        # Combine results for this batch

        for r_spine, r_neck, r_head  in zip(spine_results, neck_results, head_results ):
            tile_results.append({
                'label': r_spine['ID'],
                #'dendrite_id': int(dend_id),
                'start_coords': r_spine['start_coords'],
                'spine_area': r_spine['spine_area'],
                'spine_vol_m': r_spine['spine_volume'],
                'spine_sa_m': r_spine['spine_surface_area'],
                'spine_length': r_spine['spine_length'],
                'head_area': r_head['head_area'],
                'head_vol_m': r_head['head_volume'],
                'head_sa_m': r_head['head_surface_area'],
                'head_length': r_head['head_length'],
                # 'head_width_min': r_head['head_min_width'],
                # 'head_width_max': r_head['head_max_width'],
                'head_width_mean_m': r_head['head_mean_width'],
                'neck_area': r_neck['neck_area'],
                'neck_vol_m': r_neck['neck_volume'],
                'neck_sa_m': r_neck['neck_surface_area'],
                'neck_length': r_neck['neck_length'],
                'neck_width_min_m': r_neck['neck_min_width'],
                'neck_width_max_m': r_neck['neck_max_width'],
                'neck_width_mean_m': r_neck['neck_mean_width']

                # 'head_length_calc': r_spine['spine_length'] - r_neck['neck_length'],
                # 'head_vol_calc': r_spine['spine_volume'] - r_neck['neck_volume']
            })

    return tile_results

def analyze_spines_batch_tiled(spine_labels_vol, head_labels_vol, dendrite, neuron, locations, settings, logger,
                               scaling):
    start_time = time.time()
    results = []
    shape = spine_labels_vol.shape
    tile_size = calculate_tile_size(shape, settings, logger)
    overlap = (10, 50, 50)  # Adjust based on your needs

    # Calculate total number of tiles
    total_tiles = math.ceil(shape[0] / (tile_size[0] - overlap[0])) * \
                  math.ceil(shape[1] / (tile_size[1] - overlap[1])) * \
                  math.ceil(shape[2] / (tile_size[2] - overlap[2]))

    current_tile = 0

    for z in range(0, shape[0], tile_size[0] - overlap[0]):
        for y in range(0, shape[1], tile_size[1] - overlap[1]):
            for x in range(0, shape[2], tile_size[2] - overlap[2]):
                current_tile += 1
                z_end = min(z + tile_size[0], shape[0])
                y_end = min(y + tile_size[1], shape[1])
                x_end = min(x + tile_size[2], shape[2])

                tile_spine = spine_labels_vol[z:z_end, y:y_end, x:x_end]
                tile_head = head_labels_vol[z:z_end, y:y_end, x:x_end]
                tile_dendrite = dendrite[z:z_end, y:y_end, x:x_end]
                tile_neuron = neuron[z:z_end, y:y_end, x:x_end]


                logger.info(f"      Processing tile {current_tile} of {total_tiles}: z={z}:{z_end}, y={y}:{y_end}, x={x}:{x_end}")


                # Process the tile
                tile_results = process_spine_batch_tile(f'tile{z}_{y}_{x}', tile_spine, tile_head, tile_dendrite, tile_neuron, locations, settings,
                                            logger, scaling)

                # Adjust coordinates to global space
                for result in tile_results:
                    result['start_coords'] = [coord + offset for coord, offset in
                                              zip(result['start_coords'], [z, y, x])]

                results.extend(tile_results)
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    spine_results_df = pd.DataFrame(results)
    spine_results_df = filter_duplicated_spine_results(spine_results_df)
    logger.info(f"      Total time taken for tiled spine mesh analysis: {time.time() - start_time:.2f} seconds")
    return spine_results_df


def filter_duplicated_spine_results(df):
    print("Diagnostic information:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Index name: {df.index.name}")
    print(f"Index type: {type(df.index)}")
    print("Data types of columns:")
    print(df.dtypes)
    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    # Check if 'label' is already the index
    if df.index.name == 'label':
        df = df.reset_index()

    # Ensure 'label' is a column
    if 'label' not in df.columns:
        raise ValueError("'label' is neither a column nor the index of the DataFrame")

        # Convert the 'label' column to strings
    df['label'] = df['label'].astype(str)

    # Create a temporary column for sorting
    df['temp_sort'] = df['label'].astype(int)

    # Group by 'label' and keep the row with the largest spine_vol_m for each group
    df_final = df.loc[df.groupby('label')['spine_vol_m'].idxmax()]

    # Sort the final dataframe by the temporary sort column
    df_final = df_final.sort_values('temp_sort')

    # Remove the temporary sort column
    df_final = df_final.drop('temp_sort', axis=1)

    print(f"Original shape: {df.shape}")
    print(f"Final shape after deduplication: {df_final.shape}")


    return df_final


def analyze_spines_batch(spine_labels_vol, head_labels_vol, dendrite, neuron, locations, settings, logger, scaling):
    #dendrites and labels and must be binarized for further analysis

    start_time = time.time()
    #free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    #logger.info(f"       Available GPU memory: {free_mem / 1e9:.2f} GB")
    mem_required, free_mem, total_mem = check_gpu_memory_requirements(spine_labels_vol, logger)
    # estimate requirements

    if mem_required > 0.25 * total_mem:
        logger.info("       Volume exceeds 25% of GPU memory. Implementing tiling strategy.")

        return analyze_spines_batch_tiled(spine_labels_vol, head_labels_vol, dendrite, neuron, locations, settings,
                                          logger, scaling)

    # Move data to GPU and create multi-channel array
    cp_multi_channel = cp.stack([cp.asarray(spine_labels_vol),
                                cp.asarray(head_labels_vol),
                                cp.asarray(dendrite),
                                cp.asarray(neuron)], axis=-1)
    logger.info(f"       Multi-channel GPU array created.")
    # Get unique labels from array A (excluding background)
    labels = cp.unique(cp_multi_channel[:, :, :, 0])
    labels = labels[labels != 0]

    avg_sub_volume_shape = 6 / scaling[0], 6 / scaling[1], 6 / scaling[2]  # spine or neck
    dtype_size = 2  # 4 bytes per float32
    batch_size = calculate_batch_size(mem_required, spine_labels_vol, avg_sub_volume_shape, dtype_size)
    # print(f"Using batch size: {batch_size}")
    logger.info(f"       Using batch size of {batch_size} spines...")

    results = []

    #batch
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        logger.info(f"        Processing batch {i // batch_size + 1} of {len(labels) // batch_size + 1}...")

        # Extract subvolumes
        sub_volumes, start_coords = extract_subvolumes_mulitchannel_GPU_batch(cp_multi_channel, batch_labels)

        #logger shape
        if settings.additional_logging == True:
            logger.info(f"Subvolumes shape: {len(sub_volumes)}")
            #log dimensions of subvolumes0
            logger.info(f"Subvolume 0 shape: {sub_volumes[0].shape}")

        #mask subvolumes
        #logger.info(f"      Masking subvolumes...")
        sub_volumes = batch_mask_subvolumes_cp(sub_volumes, batch_labels)

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after masking: {sub_volumes[0].shape}")


        # Pad subvolumes to allow batching
        #padded_subvolumes, pad_amounts = pad_subvolumes(sub_volumes, logger)

        padded_subvolumes, pad_amounts = pad_and_center_subvolumes(sub_volumes, settings, logger)

        #Save spine arrays - shape is M
        if settings.additional_logging == True:
            logger.info(f"      Saving spine arrays...")
        #MZYXC to MZCYX for imagej
        export_subvols = cp.transpose(cp.stack(padded_subvolumes, axis=0), (0, 1, 4, 2,3 ))
        export_subvols[:,:, :3] = export_subvols[:, :, :3] * 65535 / 2
        export_subvols[:, :, 0] = export_subvols[:, :, 0] - export_subvols[:, :, 1]
        export_subvols = cp.asnumpy(export_subvols)
        #logger.info(export_subvols.shape)

        imwrite(f'{locations.arrays}/Spine_vols_{settings.filename}_b{i // batch_size + 1}.tif', export_subvols.astype(np.uint16), compression=('zlib', 1), imagej=True,
                photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX', 'mode': 'composite'},
                resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        imwrite(f'{locations.arrays}/Spine_MIPs_{settings.filename}_b{i // batch_size + 1}.tif', np.max(export_subvols, axis=1).astype(np.uint16), compression=('zlib', 1), imagej=True,
                photometric='minisblack',
                metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        #imwrite(f'{locations.arrays}/Spine_slices_{settings.filename}_b{i // batch_size + 1}.tif',
        #        export_subvols[:,export_subvols.shape[1] // 2].astype(np.uint16), imagej=True,
        #        photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
        #        resolution=(1 / settings.input_resXY, 1 / settings.input_resXY))

        del export_subvols

        if settings.additional_logging == True:
            logger.info(f"Subvolume 0 shape after padding: {padded_subvolumes[0].shape}")


        #loop over sub_volumes but also provide a counter
        #if settings.save_intermediate_data == True:
        #    for subvol, start_coord, label in zip(sub_volumes, start_coords, batch_labels):

                #save as tif and resahpe for imageJ
        #        imwrite_filename = os.path.join(locations.Meshes, f"subvol_{label}.tif")
                #imageJ ZCYX - currently ZYXC so fix

                #subvol_out = subvol.get()
        #        imwrite(imwrite_filename, subvol.get().transpose(0, 3, 1, 2).astype(np.uint16), imagej=True,
        #                photometric='minisblack', metadata={'unit': 'um', 'axes': 'ZCYX'})


        #generate data for spine
        batch_data = cp.stack([subvol[:, :, :, 0].astype(cp.float32) for subvol in padded_subvolumes])  # spine volumes
        batch_dendrite = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes]) #dendrite
        #binary of batch_dendrite
        batch_binary = cp.stack([subvol[:, :, :, 2] > 0 for subvol in padded_subvolumes])  # dendrite binary

        # calculate closest points
        closest_points = find_closest_points_batch(batch_data, batch_binary)

        #updated to remove dendrite id calc as more efficient during geodesic dist analysis
        #dendrite_ids = find_closest_dendrite_id(closest_points, batch_dendrite)
        #print(dendrite_ids)
        if settings.additional_logging == True:
            logger.info(f"batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")
            logger.info(f"batch_dendrite dtype: {batch_dendrite.dtype}, shape: {batch_dendrite.shape}")
            logger.info(f"closest_points dtype: {closest_points.dtype}, shape: {closest_points.shape}")
           # logger.info(f"dendrite_ids dtype: {dendrite_ids.dtype}, shape: {dendrite_ids.shape}")

        # Process spines and pass midlines on for neck and head analysis
        t0 = time.time()

        spine_results, midlines = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 0, True, None, True, True,"spine", locations, settings, logger)
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of spines: {time.time() - t0:.2f} seconds")


        # Subtract heads from spines and measure necks and ensure float for marching_cubes
        batch_data = cp.stack([
            cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0).astype(cp.float32)
            for subvol in padded_subvolumes
        ])

        #for subvol in sub_volumes:
        #    subvol[:, :, :, 0] = cp.maximum(subvol[:, :, :, 0] - subvol[:, :, :, 1], 0)
        t0 = time.time()

        neck_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 0, False, midlines, True, True, "neck", locations, settings,  logger) #True for min max mean width
        if settings.additional_logging == True:
            logger.info(f"      Time taken for mesh measurements of necks: {time.time() - t0:.2f} seconds")

        #now process heads
        t0 = time.time()
        batch_data = cp.stack([subvol[:, :, :, 1] for subvol in padded_subvolumes])  # spine vols
        # Process just heads - but don't need vol for length just surface area
        head_results, _ = batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, batch_labels, scaling, 1, False, midlines, True, True,"head", locations, settings, logger) #False for width using bounding box
        if settings.additional_logging == True:
                logger.info(f"      Time taken for mesh measurements of spine heads: {time.time() - t0:.2f} seconds")
        # Combine results for this batch

        for r_spine, r_neck, r_head in zip(spine_results, neck_results, head_results):
            results.append({
                'label': r_spine['ID'],
                #'dendrite_id': int(dend_id),
                'start_coords': r_spine['start_coords'],
                'spine_area': r_spine['spine_area'],
                'spine_vol_m': r_spine['spine_volume'],
                'spine_sa_m': r_spine['spine_surface_area'],
                'spine_length': r_spine['spine_length'],
                'head_area': r_head['head_area'],
                'head_vol_m': r_head['head_volume'],
                'head_sa_m': r_head['head_surface_area'],
                'head_length': r_head['head_length'],
                # 'head_width_min': r_head['head_min_width'],
                # 'head_width_max': r_head['head_max_width'],
                'head_width_mean_m': r_head['head_mean_width'],
                'neck_area': r_neck['neck_area'],
                'neck_vol_m': r_neck['neck_volume'],
                'neck_sa_m': r_neck['neck_surface_area'],
                'neck_length': r_neck['neck_length'],
                'neck_width_min_m': r_neck['neck_min_width'],
                'neck_width_max_m': r_neck['neck_max_width'],
                'neck_width_mean_m': r_neck['neck_mean_width']

                #'head_length_calc': r_spine['spine_length'] - r_neck['neck_length'],
                #'head_vol_calc': r_spine['spine_volume'] - r_neck['neck_volume']
            })
    spine_results_df = pd.DataFrame(results)

    logger.info(f"      Total time taken for mesh analysis: {time.time() - start_time:.2f} seconds")

    return spine_results_df


def find_closest_dendrite_id(closest_points, batch_dendrite):
    dendrite_ids = []

    for closest_point, dendrite in zip(closest_points, batch_dendrite):
        # Create a distance transform centered on the closest point
        point_mask = cp.zeros_like(dendrite)
        point_mask[closest_point] = 1
        dist_transform = cp.asarray(distance_transform_edt(cp.asnumpy(point_mask)))

        # Mask the distance transform with non-zero dendrite voxels
        masked_dist = cp.where(dendrite > 0, dist_transform, cp.inf)

        # Find the minimum distance point in the dendrite
        closest_dendrite_point = cp.unravel_index(cp.argmin(masked_dist), dendrite.shape)
        dendrite_id = dendrite[closest_dendrite_point]
        dendrite_ids.append(dendrite_id)

    return cp.array(dendrite_ids)

def batch_mask_subvolumes_cp(sub_volumes_list, batch_labels):
    # Convert batch_labels to a CuPy array if it's not already
    batch_labels = cp.asarray(batch_labels)

    masked_subvolumes = []

    for subvol, label in zip(sub_volumes_list, batch_labels):
        # Ensure subvol is a CuPy array (it should already be, but just in case)
        subvol = cp.asarray(subvol)

        # Reshape label to broadcast correctly
        label = label.reshape(1, 1, 1, 1)

        # Create binary masks for both channels
        mask_channel0 = (subvol[:, :, :, 0] == label)
        mask_channel1 = (subvol[:, :, :, 1] == label)

        # Create a copy of subvol to avoid modifying the original
        masked_subvol = subvol.copy()

        # Apply the masks
        masked_subvol[:, :, :, 0] = mask_channel0.astype(subvol.dtype)
        masked_subvol[:, :, :, 1] = mask_channel1.astype(subvol.dtype)

        masked_subvolumes.append(masked_subvol)

    return masked_subvolumes

def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def pad_subvolumes(subvolumes, logger):
    # Find the maximum dimensions
    max_shape = np.max([subvol.shape for subvol in subvolumes], axis=0)
    logger.info(f"       Max spine volume shape: {max_shape}")

    # Pad each subvolume to the maximum dimensions
    padded_subvolumes = []
    pad_amounts = []
    for subvol in subvolumes:
        pad_width = [(0, max_dim - dim) for max_dim, dim in zip(max_shape, subvol.shape)]
        padded_subvol = np.pad(subvol, pad_width, mode='constant', constant_values=0)
        padded_subvolumes.append(padded_subvol)
        pad_amounts.append(pad_width)

    return padded_subvolumes, pad_amounts


def pad_and_center_subvolumes(subvolumes, settings, logger):

    # Find the maximum dimensions
    max_shape = np.max([subvol.shape for subvol in subvolumes], axis=0)
    if settings.additional_logging == True:
        logger.info(f"       Max spine volume shape: {max_shape}")

    # Pad each subvolume to the maximum dimensions
    padded_subvolumes = []
    pad_amounts = []
    for subvol in subvolumes:
        # Calculate padding for each dimension
        pad_width = [(max_dim - dim) // 2 + ((max_dim - dim) % 2 > 0) for max_dim, dim in zip(max_shape, subvol.shape)]

        # Create the full pad_width tuple for np.pad
        full_pad_width = [(pad, max_dim - dim - pad) for pad, max_dim, dim in zip(pad_width, max_shape, subvol.shape)]

        padded_subvol = np.pad(subvol, full_pad_width, mode='constant', constant_values=0)
        padded_subvolumes.append(padded_subvol)
        pad_amounts.append(full_pad_width)

    return padded_subvolumes, pad_amounts

def unpad_result(result, pad_amount):
    # Adjust start_coords based on padding
    result['start_coords'] = [coord - pad for coord, (pad, _) in zip(result['start_coords'], pad_amount)]
    return result


def batch_mesh_analysis(batch_data, start_coords, closest_points, pad_amounts, object_ids, scaling, channel, calc_midline, midlines,
                        multiple_widths_option, save_meshes, prefix, locations, settings, logger):
    # Pad subvolumes to allow batching
    #subvolumes will be prepadded so and provided as the batch data cp format previously in fucntion
    #padded_subvolumes, pad_amounts = pad_subvolumes(subvolumes)

    #batch_data = cp.stack([subvol[:, :, :, channel] for subvol in padded_subvolumes])
    #batch_binary = cp.stack([subvol[:, :, :, 2] for subvol in padded_subvolumes])
    results = []

    if calc_midline:

        midlines = [None] * len(batch_data)
    else:
        closest_points = [None] * len(batch_data)
    #cp.cuda.Stream.null.synchronize()
    #cp.get_default_memory_pool().free_all_blocks()
    #cp.get_default_pinned_memory_pool().free_all_blocks()

    #cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    #cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)



    if settings.additional_logging == True:
        if cp.issubdtype(batch_data.dtype, cp.floating):
            if cp.isnan(batch_data).any() or cp.isinf(batch_data).any():
                logger.error("batch_data contains NaN or Inf values")
                # Handle the error as appropriate
        else:
            logger.info("batch_data is of integer type; skipping NaN and Inf checks")
        logger.info(f"batch_data dtype: {batch_data.dtype}, shape: {batch_data.shape}")


    # create folder f'{locations.Meshes}/{settings.filename}
    if settings.save_val_data:
        if not os.path.exists(f'{locations.Meshes}/{settings.filename}'):
            os.makedirs(f'{locations.Meshes}/{settings.filename}')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/head')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/spine')
            os.makedirs(f'{locations.Meshes}/{settings.filename}/neck')
    for i, (subvol, start_coord, closest_point, pad_amount, obj_id, midline) in enumerate(
            zip(batch_data, start_coords, closest_points, pad_amounts, object_ids, midlines)):

        if settings.additional_logging == True:
            logger.info(f"Processing object ID {obj_id}")
            logger.info(f"Type of subvol: {type(subvol)}, shape: {subvol.shape}")
            logger.info(f"Type of start_coord: {type(start_coord)}, value: {start_coord}")
            logger.info(f"Type of closest_point: {type(closest_point)}, value: {closest_point}")
            logger.info(f"Type of pad_amount: {type(pad_amount)}, value: {pad_amount}")
            logger.info(f"Type of midline: {type(midline)}")
            logger.error(f"subvol contains NaN or Inf values for object ID {obj_id}")
            logger.info(f"subvol dtype: {subvol.dtype}, shape: {subvol.shape}")

            mempool = cp.get_default_memory_pool()
            logger.info(f"GPU memory used: {mempool.used_bytes() / 1024 ** 2:.2f} MB")

        if cp.issubdtype(subvol.dtype, cp.floating):
            if cp.isnan(subvol).any() or cp.isinf(subvol).any():
                logger.error(f"subvol contains NaN or Inf values for object ID {obj_id}")
                # Handle the error as appropriate
        elif settings.additional_logging == True:
            logger.info(f"subvol is of integer type; skipping NaN and Inf checks for object ID {obj_id}")

        # Calculate voxel count

        try:
            voxel_count = int(cp.sum(subvol > 0).get())

        except Exception as e:
            logger.error(f"Error calculating voxel count for {prefix} {i} spine {obj_id}: {str(e)}")
            voxel_count = 0

        #create a MIP of the subvol and then sum the pixels
        mip = cp.max(subvol, axis=0)
        area = cp.sum(mip > 0) * scaling[1] * scaling[2]

        # add in zero values for non existant necks
        length = 0.0
        min_width = 0.0
        max_width = 0.0
        mean_width = 0.0
        #print(area)

        try:
            # Attempt to create mesh
            verts, faces, _, _ = measure.marching_cubes(subvol.get())

            if len(verts) == 0 or len(faces) == 0:
                #logger.info(f"No surface found for {prefix} {i}. Creating empty result.")
                result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
                results.append(result)
                continue

            verts = cp.asarray(verts) * cp.asarray(scaling)
            faces = cp.asarray(faces)

            volume = mesh_volume(verts, faces)
            surface_area = measure_surface_area_gpu(verts, faces)
            ## NEEDS ATTENTION
            if multiple_widths_option:
                if voxel_count == 1:
                    length, min_width, max_width, mean_width =  1*scaling[1], 1*scaling[1],1*scaling[1], 1*scaling[1]
                    #create volume by multplying voxel by scaling XYZ dims
                    volume = scaling[0] * scaling[1] * scaling[2]
                    if prefix == "neck" and i == 0:
                        print(f"Volume is {volume} for {prefix} {i}")
                if calc_midline and voxel_count > 1:
                    length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length_closest(
                        subvol, verts, closest_point, logger, scaling)
                    midlines[i] = skeleton_result
                else:
                    if midline is not None:
                        midline_voxels = cp.round(midline).astype(cp.int32)
                        mask = cp.zeros(subvol.shape, dtype=cp.bool_)
                        for voxel in midline_voxels:
                            mask[tuple(voxel)] = True
                        midline_mask = mask * (subvol > 0)
                        midline = cp.argwhere(midline_mask)
                    else:
                        #print(f"Midline is None for {prefix} {i}. Using default values.")
                        length, min_width, max_width, mean_width, skeleton_result = 0.0, 0.0, 0.0, 0.0, None

                    if midline is not None and len(midline) > 0 and voxel_count > 1:
                        try:
                            length, min_width, max_width, mean_width, skeleton_result = mesh_neck_width_and_length_midline_calculated(
                            midline, verts, logger, scaling)
                        except Exception as e:
                            logger.error(f"Unexpected error calculating with predifined midline {prefix} {i} spine {obj_id}: {str(e)}")
                            logger.info(f'volume is {volume} for {prefix} {i}')
                            #logger other values such as length width and skeleton result
                            logger.info(f"Length is {length} for {prefix} {i}")
                            logger.info(f"Min width is {min_width} for {prefix} {i}")
                            logger.info(f"Max width is {max_width} for {prefix} {i}")
                            logger.info(f"Mean width is {mean_width} for {prefix} {i}")
                            logger.info(f"Skeleton result is {skeleton_result} for {prefix} {i}")
                            #length, min_width, max_width, mean_width = 0.0, 0.0, 0.0, 0.0
                            skeleton_result = None
                    else:
                        skeleton_result = midline
                #if prefix == "neck" and i == 0:
                #    print(f"Volume is {volume} for {prefix} {i}")
                result = {
                    'ID': obj_id,
                    'start_coords': start_coord.get().tolist() if closest_point is None else (
                                start_coord + closest_point).get().tolist(),
                    f'{prefix}_area': float(area),
                    f'{prefix}_volume': float(volume),
                    f'{prefix}_surface_area': float(surface_area),
                    f'{prefix}_length': float(length),
                    f'{prefix}_min_width': float(min_width),
                    f'{prefix}_max_width': float(max_width),
                    f'{prefix}_mean_width': float(mean_width)
                }

            else:
                length, width = calculate_simple_length_and_width(verts)
                skeleton_result = None

                result = {
                    'ID': obj_id,
                    'start_coords': start_coord.get().tolist() if closest_point is None else (
                                start_coord + closest_point).get().tolist(),
                    f'{prefix}_area': float(area),
                    f'{prefix}_volume': float(volume),
                    f'{prefix}_surface_area': float(surface_area),
                    f'{prefix}_length': float(length),
                    f'{prefix}_width': float(width)
                }

            if settings.save_val_data:
                mesh_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.obj'
                save_mesh(verts.get(), faces.get(), mesh_filename)

                subvol_data = subvol.get().astype(np.uint16) * 65535

                if skeleton_result is not None:
                    skeleton_channel = np.zeros_like(subvol_data)
                    closet_point_channel = np.zeros_like(subvol_data)
                    skeleton_voxels = np.atleast_2d(skeleton_result.get())

                    for point in skeleton_voxels:
                        point = tuple(point)
                        if np.all(np.array(point) >= 0) and np.all(np.array(point) < np.array(skeleton_channel.shape)):
                            skeleton_channel[point] = 65535

                    if closest_point is not None:
                        closet_point_channel[tuple(cp.asnumpy(closest_point))] = 65535

                    multi_channel_image = np.stack([subvol_data, skeleton_channel, closet_point_channel], axis=1)
                    multi_channel_image = multi_channel_image * 65535

                    imsave_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.tif'
                    imwrite(imsave_filename, multi_channel_image.astype(np.uint16), compression=('zlib', 1), imagej=True,
                            metadata={'spacing': scaling[0], 'unit': 'um'})
                else:
                    imsave_filename = f'{locations.Meshes}/{settings.filename}/{prefix}/{obj_id}.tif'
                    imwrite(imsave_filename, subvol_data, compression=('zlib', 1), imagej=True, metadata={'spacing': scaling[0], 'unit': 'um'})

            # Unpad the coordinate
            result = unpad_result(result, pad_amount)
            results.append(result)

        except RuntimeError as e:
            if str(e) == 'No surface found at the given iso value.':
                #logger.info(f"No surface found for {prefix} {i}. Creating empty result.")
                result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
                result = unpad_result(result, pad_amount)
                results.append(result)
            else:
                logger.error(f"Error processing {prefix} {i}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing {prefix} {i} spine {obj_id}: {str(e)}")

            result = create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option)
            result = unpad_result(result, pad_amount)
            results.append(result)

    return results, midlines


def create_empty_result(obj_id, start_coord, closest_point, prefix, multiple_widths_option):
    base_result = {
        'ID': obj_id,
        'start_coords': start_coord.get().tolist() if closest_point is None else (
                    start_coord + closest_point).get().tolist(),
        f'{prefix}_volume': 0.0,
        f'{prefix}_surface_area': 0.0,
    }

    if multiple_widths_option:
        base_result.update({
            f'{prefix}_area':0.0,
            f'{prefix}_length': 0.0,
            f'{prefix}_min_width': 0.0,
            f'{prefix}_max_width': 0.0,
            f'{prefix}_mean_width': 0.0
        })
    else:
        base_result.update({
            f'{prefix}_length': 0.0,
            f'{prefix}_width': 0.0
        })

    return base_result

def preprocess_faces_for_viz(faces):
    # Add the count of vertices (3) at the start of each face
    return np.column_stack((np.full(faces.shape[0], 3), faces))


def save_mesh(verts, faces, filename):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

def calculate_simple_length_and_width(verts):
    # Calculate the overall length (max dimension in any direction)
    bbox_min = cp.min(verts, axis=0)
    bbox_max = cp.max(verts, axis=0)
    length = float(cp.max(bbox_max - bbox_min))

    # Project vertices onto XY plane
    verts_xy = verts[:, :2]

    # Calculate the centroid of the XY projection
    centroid = cp.mean(verts_xy, axis=0)

    # Calculate distances from each point to the centroid
    distances = cp.linalg.norm(verts_xy - centroid, axis=1)

    # Find the two points furthest from each other
    max_distance_idx = cp.argmax(distances)
    furthest_point = verts_xy[max_distance_idx]

    # Calculate vectors from centroid to each point
    vectors = verts_xy - centroid

    # Calculate the perpendicular direction to the longest axis
    longest_vector = furthest_point - centroid
    perpendicular = cp.array([-longest_vector[1], longest_vector[0]])
    perpendicular /= cp.linalg.norm(perpendicular)

    # Project all points onto this perpendicular direction
    projections = cp.dot(vectors, perpendicular)

    # Calculate width as the difference between max and min projections
    width = float(cp.max(projections) - cp.min(projections))


    return length, width

def find_closest_points_batch(batch_data, batch_binary):
    closest_points = []
    for data, binary in zip(batch_data, batch_binary):
        # Calculate distance transform on the binary image
        dist_transform = cp.asarray(distance_transform_edt(cp.asnumpy(binary == 0)))

        # Mask the distance transform with the object
        masked_dist = cp.where(data > 0, dist_transform, cp.inf)

        # Find the minimum distance point
        closest_point = cp.unravel_index(cp.argmin(masked_dist), data.shape)
        closest_points.append(closest_point)

    return cp.array(closest_points)



def measure_surface_area_gpu(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    area = 0.5 * cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)
    return float(cp.sum(area))


def measure_skeleton_length(points):
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points.get())
    dist, _ = tree.query(points.get(), k=len(points))
    return cp.max(dist).item()


def filter_dendrites(dendrites, settings,logger):
    # Label initial dendrites
    dendrite_labels, num_detected = ndimage.label(dendrites)

    # Calculate volumes and filter
    dend_vols = ndimage.sum_labels(dendrites, dendrite_labels, index=range(1, num_detected + 1))
    large_dendrites = dend_vols >= settings.min_dendrite_vol

    # Create new dendrite binary and relabel
    filt_dendrites, num_filtered = ndimage.label(np.isin(dendrite_labels, np.nonzero(large_dendrites)[0] + 1))

    # Free memory of temporary variables
    del dendrite_labels, dend_vols, large_dendrites

    logger.info(
        f"    Processing {num_filtered} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")

    return filt_dendrites


def calculate_dendrite_length_and_volume_fast(labeled_dendrites, skeleton, logger):
    skeletonize_time = time.time()
    skeleton_coords = np.array(np.nonzero(skeleton)).T
    coords_time = time.time()
    print(f"Coordinate extraction done in {coords_time - skeletonize_time:.2f} seconds")
    # Retrieve labels at skeleton points from the original image
    print("Retrieving labels from skeleton points...")
    labeled_skeletons = labeled_dendrites[tuple(skeleton_coords.T)]
    labels_time = time.time()
    print(f"Label retrieval done in {labels_time - coords_time:.2f} seconds")

    multi_time = time.time()
    print("multiplication approach")
    # Calculate labeled skeletons
    #labeled_skeletons2 = skeleton * labeled_dendrites
    print(f"Multiplication done in {time.time() - multi_time:.2f} seconds")

    # Get unique labels, excluding background (0)
    unique_dendrite_labels = np.unique(labeled_skeletons)
    unique_dendrite_labels = unique_dendrite_labels[unique_dendrite_labels != 0]

    dendrite_lengths = {label: 0 for label in unique_dendrite_labels}
    # Find unique labels present in the skeleton
    unique_labels_in_skeleton = np.unique(labeled_skeletons)
    unique_labels_in_skeleton = unique_labels_in_skeleton[unique_labels_in_skeleton != 0]
    #logger.info(f"Found {len(unique_labels_in_skeleton)} unique dendrite labels in skeleton.")

    # Calculate lengths only for labels present in the skeleton
    lengths = ndimage.sum(np.ones_like(labeled_skeletons), labeled_skeletons, index=unique_labels_in_skeleton)

    # Update the dendrite_lengths dictionary with calculated lengths
    for label, length in zip(unique_labels_in_skeleton, lengths):
        dendrite_lengths[label] = length

    # Calculate dendrite lengths
    #dendrite_lengths = ndimage.sum(np.ones_like(labeled_skeletons), labeled_skeletons, index=unique_dendrite_labels)
    #dendrite_lengths = {label: 0 for label in unique_dendrite_labels}

    # Calculate dendrite volumes
    # Step 6: Calculate dendrite volumes for all labels
   # dendrite_volumes = ndimage.sum(labeled_dendrites > 0, labeled_dendrites, index=unique_dendrite_labels)
    dendrite_volumes = ndimage.sum(np.ones_like(labeled_dendrites), labeled_dendrites, index=unique_dendrite_labels)
    dendrite_volumes = dict(zip(unique_dendrite_labels, dendrite_volumes))

    #dendrite_volumes = ndimage.sum(np.ones_like(labeled_dendrites), labeled_dendrites, index=unique_dendrite_labels)

    # Create dictionaries with labels as keys
    #dendrite_lengths = dict(zip(unique_dendrite_labels, dendrite_lengths))
    #dendrite_volumes = dict(zip(unique_dendrite_labels, dendrite_volumes))

    return dendrite_lengths, dendrite_volumes, skeleton_coords, labeled_skeletons


def adaptive_distance_transform(image, logger, threshold_size=20 * 2000 * 2000):
    # Ensure the input is binary
    time_initial = time.time()
    binary_image = image.astype(bool)

    # Calculate the current image size
    current_size = np.prod(binary_image.shape)

    # Calculate the scale factor
    size_ratio = current_size / threshold_size
    scale_factor = max(1, np.ceil(np.sqrt(size_ratio) * 2) / 2)

    # If scale_factor is 1, perform regular distance transform
    if scale_factor == 1:
        return ndimage.distance_transform_edt(np.invert(binary_image))

    scale_factor = scale_factor * 2
    logger.info(f"    Using scale factor {round(scale_factor,1)} to reduce computation time for distance calculation.")
    # Calculate new shape
    new_shape = tuple(int(s / scale_factor) for s in binary_image.shape)

    # Downsample the image
    small_image = resize(binary_image, new_shape, order=0, preserve_range=True, anti_aliasing=False)

    # Calculate distance transform on small image
    #logger.info("    Calculating distance transform on downsampled image...")
    small_distance = ndimage.distance_transform_edt(np.invert(small_image))

    # Scale up the distance map
    #logger.info("    Resizing distance map to original size...")
    large_distance = resize(small_distance, binary_image.shape, order=1, preserve_range=True)

    #save large distance to tif using tifffile
    #imwrite(r"C:\Users\Luke_H\Desktop\large_distance.tif", large_distance.astype(np.uint16))


    #zoom_factors = (binary_image.shape[0] / small_distance.shape[0],
    #                binary_image.shape[1] / small_distance.shape[1])
    #large_distance = ndimage.zoom(small_distance, zoom_factors, order=1, mode='nearest')
    #imwrite(r"C:\Users\Luke_H\Desktop\large_distance2.tif", large_distance.astype(np.uint16))
    # Correct the distances and round to nearest integer
    large_distance = np.round(large_distance * scale_factor).astype(int)
    logger.info(f"    Time taken for adaptive distance calculation: {time.time() - time_initial:.2f} seconds")
    return large_distance


def spine_and_whole_neuron_processing(image, labels_vol, spine_summary, settings, locations, filename, log, logger):

    time_initial = time.time()

    #log_memory_usage(logger)

    neuron = image[:,settings.neuron_channel-1,:,:]



    if neuron.size > 1e8:
        logger.info(f"   The neuron channel for this image is ~{neuron.size / 1e9:.2f} GB in size. This may take considerable time to process.")
        logger.info(f"   Based on previous experiments it may take ~{round(neuron.size * 2.5e-8,0)} minutes, if sufficient resources are available.")

        # temp size limits
    if neuron.size > 1e9:
        logger.info(
            f"    *Note, as the dataset is over 1GB, full 3D validation data export has been disabled (as these volumes can be 20x raw input)."
            f"\n    To generate 3D validation datasets, please isolate specific regions of the dataset and process separately.")
        logger.info(
            f"    Due to the size of this image volume, neck generation features are currently unavailable.")
        settings.neck_analysis = False
        #settings.mesh_analysis = True
        settings.save_val_data = False
    else:
        settings.neck_analysis = False
        settings.mesh_analysis = True

    if settings.model_type == 1:
        spines = (labels_vol == 1)
        dendrites = (labels_vol == 2)
        soma = (labels_vol==3)
    elif settings.model_type == 2:
        dendrites = (labels_vol == 1)
        soma = (labels_vol==2)
        spines = (labels_vol == 10) # create an empty volume for spines
    elif settings.model_type == 3:
        spines = (labels_vol == 1)
        dendrites = (labels_vol == 2)
        soma = (labels_vol==3)
        necks = (labels_vol == 4)

    #filter dendrites
    logger.info("   Filtering dendrites...")
    labeled_dendrites = filter_dendrites(dendrites, settings, logger)
    if np.max(labeled_dendrites) > 0:

        if np.max(soma) == 0:
            #label and filter soma
            soma_distance = soma
        else:
            logger.info("   Calculating soma distance...")
            #soma_distance = ndimage.distance_transform_edt(np.invert(soma))
            soma_distance = adaptive_distance_transform(soma, logger)

        #logger.info(f"   Processing {np.max(filt_dendrites)} dendrites...")
        #Create Distance Map
        logger.info("   Calculating distances from dendrites...")
        #dendrite_distance = ndimage.distance_transform_edt(np.invert(labeled_dendrites > 0)) #invert neuron mask to get outside distance
        dendrite_distance = adaptive_distance_transform(labeled_dendrites > 0, logger)
        logger.info("   Calculating dendrite skeleton...")
        dendrites_mask = (labeled_dendrites > 0).astype(np.uint8) #changed from dendrites
        skeleton = morphology.skeletonize(dendrites_mask)



        #if settings.save_val_data == True:
        #    save_3D_tif(neuron_distance.astype(np.uint16), locations.validation_dir+"/Neuron_Mask_Distance_3D"+file, settings)

        #Create Neuron MIP for validation - include distance map too
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, neuron_mask, soma_mask, soma_distance, skeleton, neuron_distance, density_image], locations.analyzed_images+"/Neuron/Neuron_MIP_"+file, 'float', settings)
        logger.info(f"    Time taken for initial processing: {time.time() - time_initial:.2f} seconds\n")
        #Spine Detection
        logger.info("   Analyzing spines...")
        model_options = ["Spines, Dendrites, and Soma", "Dendrites and Soma Only", "Necks, Spines, Dendrites, and Soma"]
        logger.info(f"    Using model type {model_options[settings.model_type - 1]}")
        spine_labels = spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)



        # now we have spine_labels and connected_necks for all spines for futher processing and measurements

        #logger.info(f" {np.max(spine_labels)}.")
        #max_label = np.max(spine_labels)

        #Measurements
        #Create 4D Labels
        #imwrite(locations.tables + 'Detected_spines.tif', spine_labels.astype(np.uint16), imagej=True, photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})
        time_spine = time.time()

        #spine_table, spines_filtered = spine_measurementsV2(image, spine_labels, 1, 0, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)

        spine_table, spines_filtered = initial_spine_measurements(image, spine_labels, 1, 0, settings.neuron_channel,
                                                            dendrite_distance,
                                                            settings.neuron_spine_size, settings.neuron_spine_dist,
                                                            settings, locations, filename, logger)

        logger.info(f"     Time taken for initial spine detection: {time.time() - time_spine:.2f} seconds")
        #spine_table, spines_filtered = spine_measurementsV1(image, spine_labels, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, filename, logger)

        #Create 4D Labels
        #imwrite(locations.tables + 'Detected_spines_filtered.tif', spines_filtered.astype(np.uint16), imagej=True, photometric='minisblack',
        #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX'})                                                  #soma_mask, soma_distance, )




        logger.info("\n    Calculating spine necks...")
        time_neck_connection = time.time()

        #disable neck analysis for very large datasets
        if settings.neck_analysis == False:
            logger.info(f"     Image shape is {spines_filtered.shape}. Neck analysis has been disabled.")
            connected_necks = np.zeros_like(spines_filtered)
        else:
            if settings.model_type == 3:
                neck_labels = measure.label(necks)

                # associate spines with necks
                logger.info("     Associating spines with necks...")
                neck_labels_updated = associate_spines_with_necks_gpu(spines_filtered, neck_labels, logger)
                #save this as a tif
                #imwrite(locations.Vols + 'Detected_necks.tif', neck_labels_updated.astype(np.uint16), imagej=True,
                 #           photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

                # unassociated neck voxels
                remaining_necks = (neck_labels) & (neck_labels_updated == 0)

                #imwrite(locations.Vols + 'remaining_necks.tif', remaining_necks.astype(np.uint16), imagej=True,
                #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})
                # extend spines with necks to dendrite
                # background = label == 0
                #traversable_for_necks = remaining_necks | labels == 0
                # = (remaining_necks + dendrites) == 0
                target = (dendrites_mask + (remaining_necks > 0)) - neck_labels_updated >0


                #Extend connected necks to neck on dendrite or directly to dendrite
                #currently can have paths crossing existing labels - need ensure these paths go around existing labels
                #
                logger.info("     Extending necks to dendrites...")
                extended_necks = extend_objects_GPU(neck_labels_updated, target, neuron, settings, locations,
                                   logger)
                extended_necks = np.where(neck_labels_updated == 0, extended_necks, 0)
                #imwrite(locations.Vols + 'extended_necks.tif', extended_necks.astype(np.uint16), imagej=True,
                #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

                # associate extended_necks with  remaining_necks
                logger.info("     Associating extended necks with remaining necks...")
                connected_necks = associate_spines_with_necks_gpu((extended_necks+neck_labels_updated), remaining_necks, logger)

                connected_necks = connected_necks - spines_filtered
                connected_necks = np.where(dendrites_mask == 0, connected_necks,  0)
                #imwrite(locations.Vols + 'connected_necks.tif', connected_necks.astype(np.uint16), imagej=True,
                #        photometric='minisblack', metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX'})

                # extend spines without necks
                #spines_without_necks = spine_labels.copy()
                #spines_without_necks[extended_necks > 0] = 0  # Remove spines that have necks

                # Update traversable mask to include extended B objects
                # traversable_for_necks = remaining_necks | dendrites
                #extended_spines = extend_objects(spines_without_necks, dendrites, traversable_for_necks)
                #extended_spines = extend_objects_GPU(spines_filtered, dendrites, neuron, settings, locations,
                #                   logger)
                # Combine extended A and B objects
                #connected_necks = np.maximum(extended_spines, extended_necks)

            else:
                #traversable_for_necks = dendrites == 0
                logger.info("     Extending spines to dendrites...")
                connected_necks = extend_objects_GPU(spines_filtered, dendrites_mask, neuron, settings, locations,
                                                     logger)

        # time for neck connection
        logger.info(f"     Time taken for neck generation: {time.time() - time_neck_connection:.2f} seconds")
        #logger.info(connected_necks.shape)
        # Remove spines that are connected to necks
        #connected_necks[spine_labels > 0] = 0

        #Second pass analysis
        #take spines(necks and spines filtered) create subvolumes, mask with a 5 pixel dilation and pass through the 2nd pass model
        if settings.second_pass:
            logger.info("     Running spine refinement...")
            if settings.refinement_model_path != None:
                spines_filtered, connected_necks =  second_pass_annotation(spines_filtered, connected_necks, dendrites_mask, neuron,
                                                                   locations, settings, logger)
            else:
                logger.info("Spine refinement model not found. Skipping second pass annotation. Please update location of second pass model in settings file to enable second pass annotation.")

        #log_memory_usage(logger)


        #calulating dendrite statistics ###### CLEAN UP WHOLE NEURON AND KEEP DEND STATS
        #logger.info("\n    Calculating dendrite statistics...")
        #originally whole neuron stats but now calculating dendrite specific

        logger.info("     Calculating neuron statistics...")
        neuron_length = np.sum(skeleton == 1)
        neuron_volume = np.sum(dendrites_mask ==1)

        del dendrites_mask

        logger.info("     Calculating dendrite statistics...")
        #get dendrite lengths and volumes as dictoinaries
        dendrite_lengths, dendrite_volumes, skeleton_coords, labeled_skeletons = calculate_dendrite_length_and_volume_fast(labeled_dendrites, skeleton, logger)



        #finished calculating dendrite statistics
        logger.info("      Complete.")

        logger.info("     Calculating spine dendrite ID and geodesic distance...")
        if np.max(soma) == 0:
            spine_dendID_and_geodist = calculate_dend_ID_and_geo_distance(labeled_dendrites, spines_filtered, skeleton_coords, labeled_skeletons,
                                             filename, locations, soma_vol=None)
        else:
            spine_dendID_and_geodist = calculate_dend_ID_and_geo_distance(labeled_dendrites, spines_filtered,
                                                                          skeleton_coords, labeled_skeletons,
                                                                          filename, locations, soma_vol=soma)

        #save spine dendrite ID and geodesic distance as csv using pands
        spine_dendID_and_geodist.to_csv(locations.tables + 'Detected_spines_dendrite_ID_and_geodesic_distance_' + filename + '.csv', index=False)

        logger.info("      Complete.")

        #analyze whole spines
        logger.info("\n     Analyzing spines using mesh generation in batches on GPU...")

        #combine connected_necks and Spines_filtered
        #print max id for spines fileterd and connected_necks
        if settings.additional_logging:
            logger.info(f"Max ID for spines filtered is {np.max(spines_filtered)}")
            logger.info(f"Max ID for connected necks is {np.max(connected_necks)}")


        if settings.mesh_analysis == True:
            spine_mesh_results = analyze_spines_batch(((connected_necks*~(spines_filtered>0))+spines_filtered), spines_filtered, labeled_dendrites, neuron, locations, settings, logger, [settings.input_resZ,  settings.input_resXY,  settings.input_resXY])
        #spine_mesh_results.to_csv(locations.tables + 'Detected_spines_mesh_measurements_' + filename + '.csv', index=False)

        #analyze spine necks
        #logger.info("      Analyzing spine necks...")
        #neck_results = analyze_spine_necks_batch(connected_necks, logger, [settings.input_resZ,  settings.input_resXY,  settings.input_resXY])
        #neck_results.to_csv(locations.tables + 'Detected_necks_mesh_measurements_' + filename + '.csv', index=False)


        if len(spine_table) == 0:
            logger.info(f"  *No spines were analyzed for this image")

        else:

            logger.info("    Saving validation MIP image...")
            neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines, spines_filtered, connected_necks, labeled_dendrites, skeleton, dendrite_distance], locations.MIPs+"MIP_"+filename, 'float', settings)

            if settings.save_val_data == True:
                logger.info("    Saving validation volume image...")
                create_and_save_multichannel_tiff([neuron, spines, spines_filtered, connected_necks, labeled_dendrites, skeleton, dendrite_distance], locations.Vols+filename, 'float', settings)

            del neuron, spines, labeled_dendrites, skeleton
            gc.collect()
            #logger.info("\n   Creating spine arrays on GPU...")
            #Extract MIPs for each spine

            #spine_MIPs, spine_slices, spine_vols = create_spine_arrays_in_blocks(image, labels_vol, spines_filtered, spine_table, settings.roi_volume_size, settings, locations, filename,  logger, settings.GPU_block_size)


            ##### We now have refined labels for all spines - so we should remeasure intensities and any voxel measurements

            #perform final vox based measurements on spines for morophology and intensity
            logger.info("\n    Calculating final measurements...")
            logger.info("     Calculating final spine head measurements...")
            spine_head_table, spines_filtered = spine_vox_measurements(image, spines_filtered, 1, 0,
                                                                       settings.neuron_channel, 'head',
                                                                dendrite_distance, soma_distance,
                                                                settings.neuron_spine_size, settings.neuron_spine_dist,
                                                                settings, locations, filename, logger)

            logger.info("     Calculating final neck measurements...")
            #now measure in necks (what about if neck label doesn't exist (ensure has value 0)
            neck_table, spines_filtered = spine_vox_measurements(image, connected_necks, 1, 0,
                                                                       settings.neuron_channel, 'neck',
                                                                       dendrite_distance, soma_distance,
                                                                       settings.neuron_spine_size,
                                                                       settings.neuron_spine_dist,
                                                                       settings, locations, filename, logger)

            #merge
            del connected_necks, dendrite_distance, soma_distance, spines_filtered
            gc.collect()

            #multiply geodesic_distance by settings.input_resXY to get in microns
            spine_dendID_and_geodist['geodesic_dist'] = spine_dendID_and_geodist['geodesic_dist'] * settings.input_resXY

            spine_table = merge_spine_measurements(spine_table, spine_dendID_and_geodist, settings, logger)

            spine_table = merge_spine_measurements(spine_table, spine_head_table, settings, logger)
            spine_table = merge_spine_measurements(spine_table, neck_table, settings, logger)

            if settings.mesh_analysis == True:
            #drop 'start_coords' from spine_mesh_results
                spine_mesh_results.drop(['start_coords'], axis=1, inplace=True)

                if settings.additional_logging:
                    pd.set_option('display.max_columns', None)
                    logger.info(spine_table.columns)
                    logger.info(spine_mesh_results.columns)
                    logger.info(spine_table.head())
                    logger.info(spine_mesh_results.head())

            #save these dfs as csvs
            #spine_table.to_csv(locations.tables + 'Detected_spines_vox_measurements_' + filename + '.csv', index=False)
            #spine_mesh_results.to_csv(locations.tables + 'Detected_spines_mesh_measurements_' + filename + '.csv', index=False)

                spine_table = merge_spine_measurements(spine_table, spine_mesh_results, settings, logger)




            logger.info("     Calculating final complete spine measurements...")
            spine_table.insert(4, f'spine_vol',
                              spine_table['head_vol'] + spine_table['neck_vol'])
            spine_table.insert(9, f'spine_C1_int_density',
                               spine_table['head_C1_int_density'] + spine_table['neck_C1_int_density'])
            spine_table.insert(10, f'spine_C1_max_int', np.maximum(spine_table['head_C1_max_int'], spine_table['neck_C1_max_int']))
            spine_table.insert(11, f'spine_C1_mean_int', spine_table['spine_C1_int_density']/ spine_table['spine_vol'])

            spine_table = move_column(spine_table, 'neck_vol', 6)

            if settings.mesh_analysis == True:
                #adjust columns for mesh data
                spine_table = move_column(spine_table, 'dendrite_id', 4)

                spine_table = move_column(spine_table, 'spine_area', 4)
                spine_table = move_column(spine_table, 'spine_vol', 5)
                spine_table = move_column(spine_table, 'spine_vol_m', 6)
                spine_table = move_column(spine_table, 'spine_sa_m', 7)
                spine_table = move_column(spine_table, 'spine_length', 8)

                spine_table = move_column(spine_table, 'head_area', 9)
                spine_table = move_column(spine_table, 'head_vol', 10)
                spine_table = move_column(spine_table, 'head_vol_m', 11)
                spine_table = move_column(spine_table, 'head_sa_m', 12)
                spine_table = move_column(spine_table, 'head_length', 13)
                spine_table = move_column(spine_table, 'head_width_mean_m', 14)
                spine_table = move_column(spine_table, 'neck_area', 15)
                spine_table = move_column(spine_table, 'neck_vol', 16)
                spine_table = move_column(spine_table, 'neck_vol_m', 17)
                spine_table = move_column(spine_table, 'neck_sa_m', 18)
                spine_table = move_column(spine_table, 'neck_length', 19)
                spine_table = move_column(spine_table, 'neck_width_mean_m', 20)
                spine_table = move_column(spine_table, 'neck_width_min_m', 21)
                spine_table = move_column(spine_table, 'neck_width_max_m', 22)
                spine_table = move_column(spine_table, 'dendrite_id', 4)
                spine_table = move_column(spine_table, 'geodesic_dist', 5)

            # Loop for C2 to C5 measurements
            for i in range(2, 6):
                c_label = f'C{i}'
                head_col = f'head_{c_label}_mean_int'
                neck_col = f'neck_{c_label}_mean_int'

                if head_col in spine_table.columns and neck_col in spine_table.columns:
                    #logger.info(f"    Creating columns for {c_label} measurements...")

                    # Calculate integrated density
                    spine_table.insert(len(spine_table.columns), f'spine_{c_label}_int_density',
                                       spine_table[f'head_{c_label}_int_density'] + spine_table[
                                           f'neck_{c_label}_int_density'])

                    # Calculate max intensity
                    spine_table.insert(len(spine_table.columns),  f'spine_{c_label}_max_int',
                                       np.maximum(spine_table[f'head_{c_label}_max_int'],  spine_table[f'neck_{c_label}_max_int']))
                    #spine_table.insert(len(spine_table.columns), f'spine_{c_label}_max_int',
                     #                  spine_table[[f'head_{c_label}_max_int', f'neck_{c_label}_max_int']].np.maximum(axis=1))

                    # Calculate mean intensity
                    spine_table.insert(len(spine_table.columns), f'spine_{c_label}_mean_int',
                                       spine_table[f'spine_{c_label}_int_density'] / spine_table['spine_vol'])

                    #logger.info(f"    Columns for {c_label} measurements created successfully.")

            '''
            #use the spine_MIPs to measure spine head area - move this to the mesh section to get areas for neck head and spine
            label_areas = spine_MIPs[:, 1, :, :]
            spine_areas = np.sum(label_areas > 0, axis=(1, 2))
            spine_masks = label_areas > 0
            spine_ids = np.nan_to_num(np.sum(label_areas * spine_masks, axis=(1, 2)) / np.sum(spine_masks, axis=(1, 2), where=spine_masks))

            df_spine_areas = pd.DataFrame({'head_area_v': spine_areas})
            df_spine_areas['label'] = spine_ids
            
            spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
            spine_table.insert(5, 'head_area', spine_table['head_area_v'] * (settings.input_resXY **2))
            spine_table.drop(['head_area_v'], axis=1, inplace=True)
            '''


            #df_spine_areas['label'] = spine_table['label'].values
            # Reindex df_spine_areas to match the index of spine_table
            #df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
            #df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False) 
            #spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts
            #spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])






            #spine_table.drop(['dendrite_id'], axis=1, inplace=True)
            #update label column to id
            spine_table.rename(columns={'label': 'spine_id'}, inplace=True)

            #drop some metrics that need furth optimization width measuremnts
            drop_columns = ['head_width_mean_m', 'neck_width_mean_m', 'neck_width_min_m', 'neck_width_max_m']
            spine_table.drop(columns=drop_columns, inplace=True, errors='ignore')


            spine_table.to_csv(locations.tables + filename+'_detected_spines.csv',index=False)


            create_spine_summary_dendrite(spine_table, filename, dendrite_lengths, dendrite_volumes, settings, locations)

            #create summary
            summary = create_spine_summary_neuron(spine_table, filename, neuron_length, neuron_volume, settings)

            # Append to the overall summary DataFrame
            spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
    else:
        logger.info("  *No dendrites were analyzed for this image.")
    logger.info("     Complete.\n")
    logger.info(f" Processing complete for file {filename}\n---")

    return spine_summary


def merge_spine_measurements(df1, df2, settings, logger):
    try:
        if settings.additional_logging:
            logger.info("Initial DataFrame info:")
            print_df_info_merge(df1, "df1", logger)
            print_df_info_merge(df2, "df2", logger)

        # Ensure 'label' is a column in both DataFrames
        if 'label' not in df1.columns:
            df1 = df1.reset_index()
        if 'label' not in df2.columns:
            df2 = df2.reset_index()

        # Convert 'label' to string in both DataFrames
        df1['label'] = df1['label'].astype(str)
        df2['label'] = df2['label'].astype(str)

        # Convert any numpy arrays to lists
        df1 = df1.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df2 = df2.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # Merge the DataFrames on the 'label' column
        merged_df = pd.merge(df1, df2, on='label', how='outer', suffixes=('_1', '_2'))

        if settings.additional_logging:
            logger.info("Merged DataFrame info:")
            print_df_info_merge(merged_df, "merged_df", logger)

        # Combine columns with _1 and _2 suffixes
        for col in merged_df.columns:
            if col.endswith('_1'):
                base_col = col[:-2]
                col_2 = base_col + '_2'
                if col_2 in merged_df.columns:
                    merged_df[base_col] = merged_df[col].combine_first(merged_df[col_2])
                    merged_df = merged_df.drop(columns=[col, col_2])
                else:
                    merged_df = merged_df.rename(columns={col: base_col})
            elif col.endswith('_2') and col[:-2] not in merged_df.columns:
                merged_df = merged_df.rename(columns={col: col[:-2]})

        # Fill NaN values with 0
        merged_df = merged_df.fillna(0)

        # Convert 'label' back to int if possible
        try:
            merged_df['label'] = merged_df['label'].astype(int)
        except ValueError:
            logger.warning("Could not convert 'label' back to int. Keeping as string.")

        if settings.additional_logging:
            logger.info("Final merged DataFrame info:")
            print_df_info_merge(merged_df, "final_merged_df", logger)

        return merged_df

    except Exception as e:
        logger.error(f"An error occurred in merge_spine_measurements: {str(e)}")
        logger.info("Printing additional information about the DataFrames:")
        print_df_info_merge(df1, "df1", logger)
        print_df_info_merge(df2, "df2", logger)
        raise  # Re-raise the exception after printing debug info


def print_df_info(df, name):
    print(f"DataFrame {name} info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nAny columns with object dtype:")
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        print(f"Column '{col}' unique values:")
        print(df[col].unique())
    print("\n")

def print_df_info_merge(df, name, logger):
    logger.info(f"Info for {name}:")
    logger.info(f"Shape: {df.shape}")
    logger.info("Columns:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")
    logger.info(f"Index: {df.index}")
    logger.info(f"Index type: {type(df.index)}")
    logger.info("First few rows:")
    logger.info(df.head())
    logger.info("\n")


def create_kdtree_from_skeleton(skeleton_coords, skeleton_labels, sampling_method='systematic', sampling_param=4):

    print(f'Skeleton points: {skeleton_coords.shape[0]}, Skeleton labels: {skeleton_labels.shape[0]}')

    # Optional: Downsample the skeleton points if needed
    if sampling_method == 'random':
        sampling_param = 0.25
        skeleton_coords, skeleton_labels = downsample_skeleton_points_random(
            skeleton_coords, skeleton_labels, sampling_rate=sampling_param)
    elif sampling_method == 'systematic':
        skeleton_coords, skeleton_labels = downsample_skeleton_points_systematic(
            skeleton_coords, skeleton_labels, step=sampling_param)
    elif sampling_method == 'voxel_grid':
        sampling_param = 2
        skeleton_coords, skeleton_labels = voxel_grid_filter(
            skeleton_coords, skeleton_labels, voxel_size=sampling_param)
    elif sampling_method is not None:
        raise ValueError('Invalid sampling method')
    # If sampling_method is None, skip downsampling

    print(f'Final skeleton points after downsampling: {skeleton_coords.shape[0]}')
    # Build the KDTree
    print("Building KDTree...")
    start_tree_time = time.time()
    tree = cKDTree(skeleton_coords)
    tree_time = time.time()
    print(f"KDTree built in {tree_time - start_tree_time:.2f} seconds")
    print(f'Tree data: {tree.data.shape}')
    return tree, skeleton_labels


def downsample_skeleton_points_random(skeleton_points, skeleton_labels, sampling_rate=0.25):
    num_points = skeleton_points.shape[0]
    num_samples = int(num_points * sampling_rate)
    if num_samples == 0:
        num_samples = 1  # Ensure at least one point is sampled

    indices = np.random.choice(num_points, size=num_samples, replace=False)
    sampled_points = skeleton_points[indices]
    sampled_labels = skeleton_labels[indices]
    return sampled_points, sampled_labels

def downsample_skeleton_points_systematic(skeleton_points, skeleton_labels, step=4):
    # Ensure step is at least 1
    step = max(1, step)
    sampled_points = skeleton_points[::step]
    sampled_labels = skeleton_labels[::step]
    return sampled_points, sampled_labels

def voxel_grid_filter(skeleton_points, skeleton_labels, voxel_size=2):
    # Quantize coordinates to the voxel grid
    quantized_coords = (skeleton_points // voxel_size).astype(np.int32)
    # Create a unique key for each voxel
    keys = quantized_coords.view([('', quantized_coords.dtype)] * quantized_coords.shape[1])
    # Find unique voxels and their indices
    _, unique_indices = np.unique(keys, return_index=True)
    sampled_points = skeleton_points[unique_indices]
    sampled_labels = skeleton_labels[unique_indices]
    return sampled_points, sampled_labels

def match_and_relabel_objects_geo(tree_A, labels_A, image_B, max_distance=None):
    #from scipy.ndimage import label, generate_binary_structure, measurements

    # Label the connected components in image_B
    #s = generate_binary_structure(3, 1)
    #labeled_B, num_features_B = label(image_B, structure=s)

    # Extract properties of labeled regions in image_B
    props_B = measure.regionprops(image_B)

    # Extract centroids and labels from props_B
    centroids_B = np.array([prop.centroid for prop in props_B])
    labels_B_1D = np.array([prop.label for prop in props_B], dtype=np.int32)

    # Perform batch KDTree query with all centroids
    distances, indices = tree_A.query(centroids_B)

    # Optionally filter by max_distance
    if max_distance is not None:
        valid = distances <= max_distance
    else:
        valid = np.ones_like(distances, dtype=bool)

    # Get the closest labels from labels_A using indices from KDTree query
    closest_labels = labels_A[indices]

    # Create a mapping array from labels in image_B to labels in image_A
   # num_features_B = image_B.max()
   # label_map_array = np.zeros(num_features_B + 1, dtype=np.int32)  # +1 because labels start from 1

    # Assign closest labels to the mapping array for valid matches
  #  label_map_array[labels_B_1D[valid]] = closest_labels[valid]

    # For invalid matches (if max_distance is specified), labels remain 0 (background)

    # Apply the mapping to relabel the entire image_B
   # relabeled_B = label_map_array[image_B]
    # Create a mapping dictionary from labels in image_B to labels in image_A
    label_map_dict = {}
    for spine_label, dendrite_label, is_valid in zip(labels_B_1D, closest_labels, valid):
        if is_valid:
            label_map_dict[spine_label] = dendrite_label
        else:
            label_map_dict[spine_label] = 0  # Background or invalid match

    # Vectorized relabeling using numpy vectorization
    #relabeled_B = np.vectorize(label_map_dict.get)(image_B)
    #relabeled_B = np.vectorize(label_map_dict.get, otypes=[np.int32])(image_B)

    # Create arrays from the mapping dictionary
    original_labels = np.array(list(label_map_dict.keys()), dtype=np.int32)
    mapped_labels = np.array(list(label_map_dict.values()), dtype=np.int32)

    # Create mapping array
    max_label_in_B = image_B.max()
    max_label_in_mapping = original_labels.max()
    mapping_array_size = max(max_label_in_B, max_label_in_mapping) + 1
    mapping_array = np.zeros(mapping_array_size, dtype=np.int32)
    mapping_array[original_labels] = mapped_labels

    # Map labels using the mapping array
    relabeled_B = mapping_array[image_B]


    # Create a DataFrame with original and updated labels
    dend_IDs = pd.DataFrame({
        'label': labels_B_1D,
        'dendrite_id': closest_labels
    })

    # Return additional data for geodesic distance calculations
    return relabeled_B, labels_B_1D, label_map_dict, indices, distances, dend_IDs


@jit(nopython=True)
def fast_unique_count(arr):
    return len(np.unique(arr))


def compute_geodesic_distance_map(object_mask, starting_points):
    # Initialize distance map with infinity
    distance_map = np.full(object_mask.shape, np.inf)

    # Set distance at starting points to zero
    for pt in starting_points:
        distance_map[pt] = 0

    # Use a queue for BFS
    from collections import deque
    queue = deque(starting_points)

    # Define neighborhood (6-connected for 3D)
    struct = generate_binary_structure(3, 1)

    while queue:
        current = queue.popleft()
        current_distance = distance_map[current]

        # Iterate over neighbors
        for offset in zip(*np.where(struct)):
            neighbor = tuple(np.array(current) + np.array(offset) - 1)
            if (0 <= neighbor[0] < object_mask.shape[0] and
                    0 <= neighbor[1] < object_mask.shape[1] and
                    0 <= neighbor[2] < object_mask.shape[2]):
                if object_mask[neighbor]:
                    if distance_map[neighbor] > current_distance + 1:
                        distance_map[neighbor] = current_distance + 1
                        queue.append(neighbor)
    return distance_map


def calculate_dend_ID_and_geo_distance(labeled_dendrites, labeled_spines, skeleton_coords, skeleton_labels, filename, locations,
                                       soma_vol=None):
    try:

        # track time of function
        t0 = time.time()
        print("Creating KDTree from skeletonized dendrites...")
        sampling_method = 'systematic'
        sampling_param = 4
        dendrite_tree, skeleton_labels = create_kdtree_from_skeleton(
            skeleton_coords, skeleton_labels, sampling_method=sampling_method, sampling_param=sampling_param)
        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} s")

        print("Matching and relabeling objects...")
        t0 = time.time()

        relabeled_spines, spine_labels_1D, spine_labels_dict, indices, distances, dend_IDs = match_and_relabel_objects_geo(
            dendrite_tree, skeleton_labels, labeled_spines)
        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} s")

        mapping_dend_to_spine_data = defaultdict(list)
        for i in range(len(spine_labels_1D)):
            spine = spine_labels_1D[i]
            dist = spine_labels_dict.get(spine, 0)
            if dist != 0:
                index_in_A = indices[i]
                coord_in_A = dendrite_tree.data[index_in_A]
                mapping_dend_to_spine_data[dist].append({
                    'label_B': spine,
                    'index_in_A': index_in_A,
                    'coord_in_A': coord_in_A,
                    'distance': distances[i],
                })

        # Initialize an array to accumulate geodesic distances
        geodesic_distance_image = np.full(labeled_dendrites.shape, np.nan, dtype=np.float32)

        print("geodesic distance")
        t0 = time.time()
        # Compute geodesic distances
        geodesic_distances_B = {}
        for dendrite, data_list in mapping_dend_to_spine_data.items():
            # Get the mask of object label_A in image_A
            dendrite_mask = (labeled_dendrites == dendrite)

            # Determine starting points
            if soma_vol is not None:
                # Find the point(s) in object A closest to an object in image_C
                object_coords = np.argwhere(dendrite_mask)
                c_coords = np.argwhere(soma_vol)
                if c_coords.size > 0:
                    from scipy.spatial import cKDTree
                    tree_C = cKDTree(c_coords)
                    distances_to_C, indices_C = tree_C.query(object_coords)
                    min_index = np.argmin(distances_to_C)
                    starting_point = tuple(object_coords[min_index])
                else:
                    # Fallback to top-left voxel
                    starting_point = tuple(object_coords[np.argmin(np.sum(object_coords, axis=1))])
            else:
                # Use the top-left voxel
                object_coords = np.argwhere(dendrite_mask)
                starting_point = tuple(object_coords[np.argmin(np.sum(object_coords, axis=1))])
            t1 = time.time()
            print(f"Time taken: {t1 - t0:.2f} s")

            print(f"Computing geodesic distance map for dendrite {dendrite}...")
            t0 = time.time()
            # Compute geodesic distance map
            distance_map = compute_geodesic_distance_map(dendrite_mask, [starting_point])
            t1 = time.time()
            print(f"Time taken: {t1 - t0:.2f} s")

            print(f"Accumulating geodesic distances for dendrite {dendrite}...")
            t0 = time.time()
            # Accumulate the geodesic distances into the image
            geodesic_distance_image[dendrite_mask] = distance_map[dendrite_mask]

            # For each data in data_list
            for data in data_list:
                label_B = data['label_B']
                coord_in_A = np.round(data['coord_in_A']).astype(int)
                coord_in_A = tuple(coord_in_A)
                geodesic_distance = distance_map[coord_in_A]
                geodesic_distances_B[label_B] = geodesic_distance

            t1 = time.time()
            print(f"Time taken: {t1 - t0:.2f} s")

        for label_B, distance in geodesic_distances_B.items():
            dend_IDs.loc[dend_IDs['label'] == label_B, 'geodesic_dist'] = distance

        #print(f"Saving relabeled image to {output_path}...")
        #save_labeled_tiff(relabeled_spines, output_path)

        # Save the geodesic distance image
        #geodesic_distance_image_path = os.path.splitext(output_path)[0] + '_geodesic_distances.tif'
        #print(f"Saving geodesic distance image to {geodesic_distance_image_path}...")
        # Replace NaNs with zeros or a large number for visualization
        geodesic_distance_image = np.nan_to_num(geodesic_distance_image, nan=0).astype(np.float32)
        #create mip of geodesic distance image
        geodesic_distance_image = np.max(geodesic_distance_image, axis=0)

        #geodesic_distance_image = geodesic_distance_image.astype(np.float32)
        # Save the geodesic distance image
        imwrite(locations.MIPs+filename+'_geodesic_dist.tif', geodesic_distance_image, compression=('zlib', 1))

       #geodesic_csv_path = os.path.splitext(output_path)[0] + '_geodesic_distances.csv'
        #with open(geodesic_csv_path, 'w', newline='') as csvfile:
         #   writer = csv.writer(csvfile)
         #   writer.writerow(['Label_B', 'Geodesic_Distance'])
         #   for label_B, distance in geodesic_distances_B.items():
         #       writer.writerow([label_B, distance])

        print(f"Original dendrite labels: {fast_unique_count(labeled_dendrites)}")
        print(f"Original spine labels: {fast_unique_count(labeled_spines)}")
        print(f"Labels after relabeling: {fast_unique_count(relabeled_spines)}")

        #print(f"Original file size of B: {os.path.getsize(image_B_path) / (1024 * 1024):.2f} MB")
        #print(f"New file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

        return dend_IDs

    except Exception as e:
        print(f"An error occurred: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise



def create_spine_summary_neuron(filtered_table, filename, dendrite_length, dendrite_volume, settings):
    #create summary table

    #spine_reduced = filtered_table.drop(columns=['label', 'z', 'y', 'x'])
    updated_table = filtered_table.iloc[:, 4:]
    #drop dendrite_id column
    updated_table = updated_table.drop(columns=['dendrite_id', 'geodesic_dist'])


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
    #spine_summary.insert(3, 'dendrite_length', dendrite_length)
    spine_summary.insert(3, 'dendrite_length', dendrite_length * settings.input_resXY)
    #spine_summary.insert(5, 'dendrite_vol', dendrite_volume)
    spine_summary.insert(4, 'dendrite_vol', dendrite_volume * settings.input_resXY*settings.input_resXY*settings.input_resZ)
    spine_summary.insert(5, 'total_spines', filtered_table.shape[0])
    spine_summary.insert(6, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length'])
    spine_summary.insert(7, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_vol'])

    return spine_summary


def create_spine_summary_dendrite(filtered_table, filename, dendrite_lengths, dendrite_volumes, settings, locations):
    #create summary table

    filtered_table = filtered_table.drop(['spine_id', 'x', 'y', 'z', 'geodesic_dist',], axis=1)

    spine_counts = filtered_table.groupby('dendrite_id').size().reset_index(name='total_spines')

    mean_metrics = filtered_table.groupby('dendrite_id').mean().reset_index()

    spine_summary = pd.merge(mean_metrics, spine_counts, on='dendrite_id')

    # update summary with additional metrics
    spine_summary.insert(0, 'Filename', filename)  # Insert a column at the beginning
    spine_summary.insert(1, 'res_XY', settings.input_resXY)
    spine_summary.insert(2, 'res_Z', settings.input_resZ)
   # spine_summary.insert(4, 'total_spines', spine_counts)

    spine_summary['dendrite_length'] = spine_summary['dendrite_id'].map(dendrite_lengths)
    spine_summary['dendrite_volume'] = spine_summary['dendrite_id'].map(dendrite_volumes)


    spine_summary.rename(columns={'avg_dendrite_length': 'dendrite_length'}, inplace=True)
    spine_summary.rename(columns={'avg_dendrite_vol': 'dendrite_vol'}, inplace=True)
    spine_summary['dendrite_length'] = spine_summary['dendrite_length'] * settings.input_resXY
    spine_summary['dendrite_volume'] = spine_summary['dendrite_volume'] *settings.input_resXY*settings.input_resXY*settings.input_resZ
    #spine_summary = move_column(spine_summary, 'dendrite_length_um', 5)
    #spine_summary = move_column(spine_summary, 'dendrite_vol_um3', 7)
    spine_summary.insert(9, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length'])
    spine_summary.insert(10, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_volume'])

    desired_order = ['Filename', 'res_XY', 'res_Z', 'dendrite_id', 'dendrite_length', 'dendrite_volume', 'total_spines', 'spines_per_um', 'spines_per_um3'] + \
                    [col for col in spine_summary.columns if col not in ['Filename', 'res_XY', 'res_Z', 'dendrite_id',
                                                                         'dendrite_length', 'dendrite_volume',
                                                                         'total_spines', 'spines_per_um',
                                                                         'spines_per_um3']] + \
                    []
    spine_summary = spine_summary[desired_order]

    spine_summary.to_csv(locations.tables + filename + '_dendrite_summary.csv', index=False)




def analyze_spines_4D(settings, locations, log, logger):
    logger.info("\nAnalyzing spines across time...")
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

        logger.info(f"  Processing{filt_dendrites} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol * settings.input_resXY *settings.input_resXY*settings.input_resZ} µm<sup>3</sup> or {settings.min_dendrite_vol} voxels...")

        if filt_dendrites > 0:

            #logger.info(f"   Processing {filt_dendrites} dendrites...")


            #Create Distance Map



            dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites_3d)) #invert neuron mask to get outside distance
            dendrite_distance_list.append(dendrite_distance)

            dendrites_3d = dendrites_3d.astype(np.uint8)

            skeleton = morphology.skeletonize(dendrites_3d)
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

            #spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_1.csv',index=False)

            label_areas = spine_MIPs[:, 1, :, :]
            spine_areas = np.sum(label_areas > 0, axis=(1, 2))

            spine_masks = label_areas > 0
            spine_ids = np.nan_to_num(np.sum(label_areas * spine_masks, axis=(1, 2)) / np.sum(spine_masks, axis=(1, 2), where=spine_masks))

            df_spine_areas = pd.DataFrame({'spine_area': spine_areas})
            df_spine_areas['label'] = spine_ids

            #df_spine_areas['label'] = spine_table['label'].values

            #df_spine_areas.to_csv(locations.tables + str(t)+ 'Detected_spines_areas.csv',index=False)
            # Reindex df_spine_areas to match the index of spine_table
            #df_spine_areas_reindex = df_spine_areas.reindex(spine_table.index)
            #df_spine_areas_reindex.to_csv(locations.tables + 'Detected_spines_'+filename+'reindex.csv',index=False)
            spine_table = spine_table.merge(df_spine_areas, on='label', how='left')
            spine_table.insert(5, 'spine_area', spine_table.pop('spine_area')) #pops and inserts
            #spine_table.insert(5, 'spine_area', df_spine_areas['spine_area'])
            spine_table.insert(6, 'spine_area_um2', spine_table['spine_area'] * (settings.input_resXY **2))

            spine_table.insert(0, 'timepoint', t+1)

            #spine_table.to_csv(locations.tables + str(t)+ 'Detected_spines_2.csv',index=False)

            #append tables
            all_spines_table = pd.concat([all_spines_table, spine_table], ignore_index=True)

            #all_spines_table.to_csv(locations.tables + str(t)+ 'Detected_spines_appened.csv',index=False)

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


    vol_over_t = all_spines_table.pivot(index='spine_id', columns='timepoint', values='spine_vol')
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
    imwrite(locations.input_dir+'/Registered/Detected_spines.tif', spines_filtered_all.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
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
    #imwrite(filename, multichannel_image, photometric='minisblack')

    imwrite(filename, multichannel_image, compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'CYX'})

    # Return the merged 2D multi-channel image as a numpy array
    return multichannel_image

def create_and_save_multichannel_tiff(images_3d, filename, bitdepth, settings):
    """
    Create TIF from a list of 3D images, merge them into a image,
    save as a 16-bit TIFF file.

    Args:
        images_3d (list of numpy.ndarray): List of 3D numpy arrays representing input images.
        filename (str): Filename for the output TIFF file.

    Returns:
        none
    """

    # Convert the list of images to a single multichannel image
    multichannel_image = np.stack(images_3d, axis=1)

    # Convert the multichannel image to 16-bit
    multichannel_image = multichannel_image.astype(np.uint16)

    imwrite(filename, multichannel_image, compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX'})


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
    #imwrite(filename, multichannel_image, photometric='minisblack')

    imwrite(filename, multichannel_image, compression=('zlib', 1), imagej=True, photometric='minisblack',
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


def initial_spine_measurements(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    """

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    logger.info("    Making initial morphology and intensity measurements for channel 1...")
    # logger.info(f" {labels.shape}, {image.shape}")
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image[:, 0, :, :],
            properties=['label', 'centroid', 'area'],  # area is volume for 3D images
        )
    )
    main_table.rename(columns={'centroid-0': 'z'}, inplace=True)
    main_table.rename(columns={'centroid-1': 'y'}, inplace=True)
    main_table.rename(columns={'centroid-2': 'x'}, inplace=True)
    # measure distance to dendrite
    logger.info("    Measuring distances to dendrite/s...")
    # logger.info(f" {labels.shape}, {dendrite_distance.shape}")
    distance_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=dendrite_distance,
            properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename distance column
    distance_table.rename(columns={'min_intensity': 'dist_to_dendrite'}, inplace=True)
    distance_table.rename(columns={'max_intensity': 'spine_length'}, inplace=True)

    distance_col = distance_table["dist_to_dendrite"]
    main_table = main_table.join(distance_col)
    distance_col = distance_table["spine_length"]
    main_table = main_table.join(distance_col)

    # filter out small objects
    volume_min = sizes[0]  # 3
    volume_max = sizes[1]  # 1500?

    # logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")

    # filter based on volume
    # logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)

    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max)]

    logger.info(f"     Total putative spines: {spinebefore}")
    logger.info(f"     Spines after volume filtering = {len(filtered_table)} ")
    # logger.info(f"  filtered table after area = {len(filtered_table)}")

    # filter based on distance to dendrite
    spinebefore = len(filtered_table)
    # logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    # logger.info(f"  filtered table before dist = {len(filtered_table)}. and distance = {dist}")
    filtered_table = filtered_table[(filtered_table['spine_length'] < dist)]
    logger.info(f"     Spines after distance filtering = {len(filtered_table)} ")

    if settings.Track != True:
        # update label numbers based on offset
        filtered_table['label'] += max_label
        labels[labels > 0] += max_label
        labels = create_filtered_labels_image(labels, filtered_table, logger)
    else:

        # Clean up label image to remove objects from image.
        ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask
        mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        labels = np.where(mask_to_keep, labels, 0)

    # update to included dendrite_id
    filtered_table.insert(4, 'dendrite_id', dendrite)

    # create vol um measurement
    filtered_table.insert(6, 'spine_vol',
                          filtered_table['area'] * (settings.input_resXY * settings.input_resXY * settings.input_resZ))
    # drop filtered_table['area']
    filtered_table = filtered_table.drop(['area'], axis=1)
    # filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)

    # create dist um cols

    filtered_table = move_column(filtered_table, 'spine_length', 7)
    # replace multiply column spine_length by settings.input_resXY
    filtered_table['spine_length'] *= settings.input_resXY
    # filtered_table.insert(8, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY))
    filtered_table = move_column(filtered_table, 'dist_to_dendrite', 9)
    filtered_table['dist_to_dendrite'] *= settings.input_resXY
    # filtered_table.insert(10, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY))
    #filtered_table = move_column(filtered_table, 'dist_to_soma', 11)
    #filtered_table['dist_to_soma'] *= settings.input_resXY
    # filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))

    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density
    #filtered_table['C1_int_density'] = filtered_table['spine_vol'] * filtered_table['C1_mean_int']

    # measure remaining channels
    #for ch in range(image.shape[1] - 1):
    #    filtered_table['C' + str(ch + 2) + '_int_density'] = filtered_table['spine_vol'] * filtered_table[
    #        'C' + str(ch + 2) + '_mean_int']

    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
    #    f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")
    #create a subset of filtered table using columns label, x, y, z
    filtered_table_subset = filtered_table[['label', 'x', 'y', 'z']]

    return filtered_table_subset, labels


def spine_vox_measurements(image, labels, dendrite, max_label, neuron_ch, prefix, dendrite_distance, soma_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)

    Returns:
        pandas table and labeled spine image
    """

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    #logger.info("    Making final morphology and intensity measurements for channel 1...")
    # logger.info(f" {labels.shape}, {image.shape}")
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image[:, 0, :, :],
            properties=['label', 'area', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename mean intensity
    main_table.rename(columns={'mean_intensity': f'{prefix}_C1_mean_int'}, inplace=True)
    main_table.rename(columns={'max_intensity': f'{prefix}_C1_max_int'}, inplace=True)
    #main_table.rename(columns={'centroid-0': 'z'}, inplace=True)
    #main_table.rename(columns={'centroid-1': 'y'}, inplace=True)
    #main_table.rename(columns={'centroid-2': 'x'}, inplace=True)

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        logger.info(f"    Measuring channel {ch + 2}...")
        # Measure
        table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:, ch + 1, :, :],
                properties=['label', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )

        # rename mean intensity
        table.rename(columns={'mean_intensity': f'{prefix}_C' + str(ch + 2) + '_mean_int'}, inplace=True)
        table.rename(columns={'max_intensity': f'{prefix}_C' + str(ch + 2) + '_max_int'}, inplace=True)

        Mean = table[f'{prefix}_C' + str(ch + 2) + '_mean_int']
        Max = table[f'{prefix}_C' + str(ch + 2) + '_max_int']

        # combine columns with main table
        main_table = main_table.join(Mean)
        main_table = main_table.join(Max)

    if prefix == 'head':
        # measure distance to dendrite
        #logger.info("    Measuring distances to dendrite...")
        # logger.info(f" {labels.shape}, {dendrite_distance.shape}")
        distance_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=dendrite_distance,
                properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )

        # rename distance column
        distance_table.rename(columns={'min_intensity': 'dist_to_dendrite_dm'}, inplace=True)
        distance_table.rename(columns={'max_intensity': 'spine_length_dm'}, inplace=True)

        distance_col = distance_table["dist_to_dendrite_dm"]
        main_table = main_table.join(distance_col)
        distance_col = distance_table["spine_length_dm"]
        main_table = main_table.join(distance_col)

        if np.max(soma_distance) > 0:
            # measure distance to dendrite
            logger.info("      Measuring distances to soma...")
            distance_table = pd.DataFrame(
                measure.regionprops_table(
                    labels,
                    intensity_image=soma_distance,
                    properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
                )
            )
            distance_table.rename(columns={'min_intensity': 'dist_to_soma'}, inplace=True)
            distance_col = distance_table["dist_to_soma"]
            main_table = main_table.join(distance_col)
        else:
            main_table['dist_to_soma'] = pd.NA

    if settings.Track != True:
        # update label numbers based on offset
        main_table['label'] += max_label
        labels[labels > 0] += max_label
        labels = create_filtered_labels_image(labels, main_table, logger)

    #else:

        # Clean up label image to remove objects from image.
        #ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask
        #mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        #labels = np.where(mask_to_keep, labels, 0)

    # update to included dendrite_id
    #filtered_table.insert(4, 'dendrite_id', dendrite)

    # create vol um measurement
    main_table.insert(1, f'{prefix}_vol', main_table['area'] * (settings.input_resXY * settings.input_resXY * settings.input_resZ))
    # drop filtered_table['area']
    main_table = main_table.drop(['area'], axis=1)
    # filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)

    # create dist um cols
    if prefix == 'head':
        main_table = move_column(main_table, 'spine_length_dm', 2)
        # replace multiply column spine_length by settings.input_resXY
        main_table['spine_length_dm'] *= settings.input_resXY
        # filtered_table.insert(8, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY))
        main_table = move_column(main_table, 'dist_to_dendrite_dm', 3)
        main_table['dist_to_dendrite_dm'] *= settings.input_resXY
        # filtered_table.insert(10, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY))
        main_table = move_column(main_table, 'dist_to_soma', 4)
        main_table['dist_to_soma'] *= settings.input_resXY
    # filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))

    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density
    main_table[f'{prefix}_C1_int_density'] = main_table[f'{prefix}_vol'] * main_table[f'{prefix}_C1_mean_int']

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        main_table[f'{prefix}_C' + str(ch + 2) + '_int_density'] = main_table[f'{prefix}_vol'] * main_table[
            f'{prefix}_C' + str(ch + 2) + '_mean_int']

    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
    #    f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")

    return main_table, labels


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

    merge_masked_mip_list = [] #Mip filtered by label - we don't need this
    merge_masked_slice_list = [] #slice filtered by label - we don't need this either

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

        #mask the spine image
        masked_image_vol = image_vol * spine_mask

        #

        # Extract 2D slice at position z
        image_slice = image_vol[volume_size_z // 2, :, :, :]
        spine_slice = spine_vol[volume_size_z // 2, :, :, :]
        label_slice = label_vol[volume_size_z // 2, :, :, :]

        masked_image_slice = masked_image_vol[volume_size_z // 2, :, :, :]
        #taking out the masked arrays - I don't think we need these        

        #why expanding -taking htis out temporarily
        ##spine_mask_expanded = cp.expand_dims(spine_mask, axis=-1)

        #extracted_volume_label_filtered = extracted_volume * spine_mask_expanded

        # Extract 2D slice before the MIP in step 3 is created
        #slice_before_mip = extracted_volume_label_filtered[volume_size_z // 2, :, :, :]
        #spine_slice = spine_mask_expanded[volume_size_z // 2, :, :, :]

        # Compute MIPs - this could be done by merging then mip but leave as this for now
        image_mip = cp.max(image_vol, axis=0)
        ##image_mip = image_mip[cp.newaxis, :, :] #expand to add label
        spine_mip=cp.max(spine_vol, axis = 0)
        ##spine_mip = spine_mip[cp.newaxis, :, :] # *65535#expand to add label
        label_mip=cp.max(label_vol, axis = 0)
        ##label_mip = label_mip[cp.newaxis, :, :]

        masked_image_mip = cp.max(masked_image_vol, axis=0)

        #logger.info(f' image mip {image_mip.shape} ,spine mip.shape {spine_mip.shape} label mip shape {label_mip.shape} ')

        merge_mip = cp.concatenate((image_mip, spine_mip, label_mip), axis=-1)

        merge_masked_mip = cp.concatenate((masked_image_mip, spine_mip, label_mip), axis=-1)


        merge_mip_list.append(merge_mip.get())

        merge_masked_mip_list.append(merge_masked_mip.get())


        ##image_slice = image_slice[cp.newaxis, :, :]#expand to add label
        ##spine_slice = spine_slice[cp.newaxis, :, :]
        ##label_slice = label_slice[cp.newaxis, :, :]


        #logger.info(f' image slice {image_slice.shape} ,spine slice.shape {spine_slice.shape} label slice shape {label_slice.shape} ')

        merge_slice = cp.concatenate((image_slice, spine_slice, label_slice), axis=-1)

        merge_masked_slice = cp.concatenate((masked_image_slice, spine_slice, label_slice), axis=-1)

        merge_slice_list.append(merge_slice.get())

        merge_masked_slice_list.append(merge_masked_slice.get())
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



        ##image_vol = cp.expand_dims(image_vol, axis=0)
        #image_vol = image_vol[cp.newaxis, :, :, :]
        ##spine_vol = cp.expand_dims(spine_vol, axis=0)
        #spine_vol = image_vol[cp.newaxis, :, :, :, :]
        ##label_vol = cp.expand_dims(label_vol, axis=0)

        #logger.info(f' image{image_vol.shape} ,spine_vol.shape {spine_vol.shape}labelvol shape {label_vol.shape} ')



        merge_vol = cp.concatenate((image_vol, spine_vol, label_vol), axis=-1)
        merge_vol_list.append(merge_vol.get())




        del image_vol, label_vol, spine_vol, image_slice, spine_slice, label_slice, image_mip, spine_mip, label_mip, merge_vol, masked_image_mip, masked_image_slice, masked_image_vol
        cp.cuda.Stream.null.synchronize()
        gc.collect()

    mip_array = np.stack(merge_mip_list)

    masked_mip_array = np.stack(merge_masked_mip_list)

    #mip_array = np.moveaxis(mip_array, 3, 1)
    ##mip_array = mip_array.squeeze(axis=0)

    slice_array = np.stack(merge_slice_list)

    masked_slice_array = np.stack(merge_masked_slice_list)

    #slice_z_array = np.moveaxis(slice_z_array, 3, 1)
    ##slice_array = slice_array.squeeze(axis=0)

    vol_array = np.stack(merge_vol_list)
    ##vol_array = vol_array.squeeze(axis=0)


    #mip_label_filtered_array = np.stack(mip_label_filtered_list)
    #mip_label_filtered_array = np.moveaxis(mip_label_filtered_array, 3, 1)
    #mip_label_filtered_array = mip_label_filtered_array.squeeze(axis=-1)

    #slice_before_mip_array = np.stack(slice_before_mip_list)
    #slice_before_mip_array = np.moveaxis(slice_before_mip_array, 3, 1)
    #slice_before_mip_array = slice_before_mip_array.squeeze(axis=-1)

    return mip_array, slice_array, masked_mip_array, masked_slice_array, vol_array

def create_spine_arrays_in_blocks(image, labels, spines_filtered, table, volume_size, settings, locations, file, logger, block_size=(50, 300, 300)):
    #suppress warning about subtracting from table without copying
    original_chained_assignment = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    smallest_axis = np.argmin(image.shape)
    image = np.moveaxis(image, smallest_axis, -1)



    mip_list = []
    slice_list = []
    masked_mip_list = []
    masked_slice_list = []
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
                print("Image shape:", image.shape)
                print("Labels shape:", labels.shape)

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

                    block_mip, block_slice, block_masked_mip, block_masked_slice, block_vol = create_filtered_and_unfiltered_spine_arrays_cupy(
                        block_image, block_spines_filtered, block_labels, block_table, volume_size, settings, locations, file, logger
                    )
                    mip_list.extend(block_mip)
                    slice_list.extend(block_slice)
                    masked_mip_list.extend(block_masked_mip)
                    masked_slice_list.extend(block_masked_slice)
                    vol_list.extend(block_vol)
                    gc.collect()

        progress_percentage = ((i + 1) * y_blocks * x_blocks) / total_blocks * 100
        #print(f'Progress: {progress_percentage:.2f}%   ', end='', flush=True)

    #print(mip_list)
    # Convert lists to arrays and concatenate
    mip_array = np.stack(mip_list, axis = 0).transpose(0, 3, 1, 2)
    slice_array = np.stack(slice_list, axis=0).transpose(0, 3, 1, 2)
    masked_mip_array = np.stack(masked_mip_list, axis = 0).transpose(0, 3, 1, 2)
    masked_slice_array = np.stack(masked_slice_list, axis=0).transpose(0, 3, 1, 2)


    vol_array = np.stack(vol_list, axis=0).transpose(0, 1, 4, 2, 3)
    print(vol_array.shape)

    #vol_array = np.transpose(vol_array, (0, 2, 1, 3, 4))
    #mip_label_filtered_array = np.stack(mip_label_filtered_list, axis=0)
    #slice_before_mip_array = np.stack(slice_before_mip_list, axis=0)
    #print(mip_array.shape)

    #logger.info(f' {mip_array.shape}')
    imwrite(locations.arrays+"/Spine_MIPs_"+file, mip_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_slices_" + file, slice_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    imwrite(locations.arrays+"/Spine_masked_MIPs_"+file, masked_mip_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_masked_slices_" + file, masked_slice_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spine_vols_" + file, vol_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'TZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #        metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
    #        resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    #imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16), imagej=True, photometric='minisblack',
    #                 metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
    #                 resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    #reenable pandas warning:
    pd.options.mode.chained_assignment = original_chained_assignment
    logger.info(f'     Complete.\n ')
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

    imwrite(locations.arrays+"/Spines_MIPs_"+file, mip_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Spines_Slices_" + file, slice_z_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
                     metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX','mode': 'composite'},
                     resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays+"/Masked_Spines_MIPs_"+file, mip_label_filtered_array.astype(np.uint16), compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))
    imwrite(locations.arrays + "/Masked_Spines_Slices_" + file, slice_before_mip_array.astype(np.uint16),  compression=('zlib', 1), imagej=True, photometric='minisblack',
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

    imwrite(locations.input_dir+"/Registered/Isolated_spines_4D.tif", final_4d.astype(np.uint16),  compression=('zlib', 1), imagej=True, photometric='minisblack',
            metadata={'unit': 'um','axes': 'TCYX', 'mode': 'composite'},
            resolution=(1/settings.input_resXY, 1/settings.input_resXY))

    return final_4d, final_4d


def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))


def spine_measurementsV2(image, labels, dendrite, max_label, neuron_ch, dendrite_distance, soma_distance, sizes, dist,
                         settings, locations, filename, logger):
    """ measures intensity of each channel, as well as distance to dendrite
    Args:
        labels (detected cells)
        settings (dictionary of settings)

    Returns:
        pandas table and labeled spine image
    """

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)

    # Measure channel 1:
    logger.info("    Making initial morphology and intensity measurements for channel 1...")
    # logger.info(f" {labels.shape}, {image.shape}")
    main_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image[:, 0, :, :],
            properties=['label', 'centroid', 'area', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename mean intensity
    main_table.rename(columns={'mean_intensity': 'C1_mean_int'}, inplace=True)
    main_table.rename(columns={'max_intensity': 'C1_max_int'}, inplace=True)
    main_table.rename(columns={'centroid-0': 'z'}, inplace=True)
    main_table.rename(columns={'centroid-1': 'y'}, inplace=True)
    main_table.rename(columns={'centroid-2': 'x'}, inplace=True)

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        logger.info(f"    Measuring channel {ch + 2}...")
        # Measure
        table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=image[:, ch + 1, :, :],
                properties=['label', 'mean_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )

        # rename mean intensity
        table.rename(columns={'mean_intensity': 'C' + str(ch + 2) + '_mean_int'}, inplace=True)
        table.rename(columns={'max_intensity': 'C' + str(ch + 2) + '_max_int'}, inplace=True)

        Mean = table['C' + str(ch + 2) + '_mean_int']
        Max = table['C' + str(ch + 2) + '_max_int']

        # combine columns with main table
        main_table = main_table.join(Mean)
        main_table = main_table.join(Max)

    # measure distance to dendrite
    logger.info("    Measuring distances to dendrite/s...")
    # logger.info(f" {labels.shape}, {dendrite_distance.shape}")
    distance_table = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=dendrite_distance,
            properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
        )
    )

    # rename distance column
    distance_table.rename(columns={'min_intensity': 'dist_to_dendrite'}, inplace=True)
    distance_table.rename(columns={'max_intensity': 'spine_length'}, inplace=True)

    distance_col = distance_table["dist_to_dendrite"]
    main_table = main_table.join(distance_col)
    distance_col = distance_table["spine_length"]
    main_table = main_table.join(distance_col)

    if np.max(soma_distance) > 0:
        # measure distance to dendrite
        logger.info("    Measuring distances to soma...")
        distance_table = pd.DataFrame(
            measure.regionprops_table(
                labels,
                intensity_image=soma_distance,
                properties=['label', 'min_intensity', 'max_intensity'],  # area is volume for 3D images
            )
        )
        distance_table.rename(columns={'min_intensity': 'dist_to_soma'}, inplace=True)
        distance_col = distance_table["dist_to_soma"]
        main_table = main_table.join(distance_col)
    else:
        main_table['dist_to_soma'] = pd.NA

    # filter out small objects
    volume_min = sizes[0]  # 3
    volume_max = sizes[1]  # 1500?

    # logger.info(f" Filtering spines between size {volume_min} and {volume_max} voxels...")

    # filter based on volume
    # logger.info(f"  filtered table before area = {len(main_table)}")
    spinebefore = len(main_table)

    filtered_table = main_table[(main_table['area'] > volume_min) & (main_table['area'] < volume_max)]

    logger.info(f"     Total putative spines: {spinebefore}")
    logger.info(f"     Spines after volume filtering = {len(filtered_table)} ")
    # logger.info(f"  filtered table after area = {len(filtered_table)}")

    # filter based on distance to dendrite
    spinebefore = len(filtered_table)
    # logger.info(f" Filtering spines less than {dist} voxels from dendrite...")
    # logger.info(f"  filtered table before dist = {len(filtered_table)}. and distance = {dist}")
    filtered_table = filtered_table[(filtered_table['spine_length'] < dist)]
    logger.info(f"     Spines after distance filtering = {len(filtered_table)} ")

    if settings.Track != True:
        # update label numbers based on offset
        filtered_table['label'] += max_label
        labels[labels > 0] += max_label
        labels = create_filtered_labels_image(labels, filtered_table, logger)
    else:

        # Clean up label image to remove objects from image.
        ids_to_keep = set(filtered_table['label'])  # Extract IDs to keep from your filtered DataFrame
        # Create a mask
        mask_to_keep = np.isin(labels, list(ids_to_keep))
        # Apply the mask: set pixels not in `ids_to_keep` to 0
        labels = np.where(mask_to_keep, labels, 0)

    # update to included dendrite_id
    filtered_table.insert(4, 'dendrite_id', dendrite)

    # create vol um measurement
    filtered_table.insert(6, 'spine_vol',
                          filtered_table['area'] * (settings.input_resXY * settings.input_resXY * settings.input_resZ))
    # drop filtered_table['area']
    filtered_table = filtered_table.drop(['area'], axis=1)
    # filtered_table.rename(columns={'area': 'spine_vol'}, inplace=True)

    # create dist um cols

    filtered_table = move_column(filtered_table, 'spine_length', 7)
    # replace multiply column spine_length by settings.input_resXY
    filtered_table['spine_length'] *= settings.input_resXY
    # filtered_table.insert(8, 'spine_length_um', filtered_table['spine_length'] * (settings.input_resXY))
    filtered_table = move_column(filtered_table, 'dist_to_dendrite', 9)
    filtered_table['dist_to_dendrite'] *= settings.input_resXY
    # filtered_table.insert(10, 'dist_to_dendrite_um', filtered_table['dist_to_dendrite'] * (settings.input_resXY))
    filtered_table = move_column(filtered_table, 'dist_to_soma', 11)
    filtered_table['dist_to_soma'] *= settings.input_resXY
    # filtered_table.insert(12, 'dist_to_soma_um', filtered_table['dist_to_soma'] * (settings.input_resXY))

    # logger.info(f"  filtered table before image filter = {len(filtered_table)}. ")
    # logger.info(f"  image labels before filter = {np.max(labels)}.")
    # integrated_density
    filtered_table['C1_int_density'] = filtered_table['spine_vol'] * filtered_table['C1_mean_int']

    # measure remaining channels
    for ch in range(image.shape[1] - 1):
        filtered_table['C' + str(ch + 2) + '_int_density'] = filtered_table['spine_vol'] * filtered_table[
            'C' + str(ch + 2) + '_mean_int']

    # Drop unwanted columns
    # filtered_table = filtered_table.drop(['spine_vol','spine_length', 'dist_to_dendrite', 'dist_to_soma'], axis=1)
    #logger.info(
     #   f"     After filtering {len(filtered_table)} spines were analyzed from a total of {len(main_table)} putative spines")

    return filtered_table, labels