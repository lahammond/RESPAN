# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
"""
Model training functions
==========


"""
__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'MIT License (see LICENSE)'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


import os
import sys
import contextlib
import random
import numpy as np
import time
import pandas as pd
import csv
import shutil
import json
from scipy.ndimage import convolve, rotate
import subprocess
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches, no_background_patches
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import img_as_uint


from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import io
import contextlib
import re
import threading



##############################################################################
# logger modules to allow training prints

def log_output(pipe, logger, log_function):
    for line in iter(pipe.readline, ''):
        log_function(line.strip())
        sys.stdout.flush()
#def log_output(pipe, logger):
 #   for line in iter(pipe.readline, b''):
  #      logger.info(line.decode().strip())
   #     sys.stdout.flush()

def run_process_with_logging(cmd, logger):
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=1)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,  bufsize=0, text=True,
                               universal_newlines=True, encoding='utf-8', errors='replace')

    # Start threads to read stdout and stderr
    #stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logger))
    #stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logger))
    stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logger, logger.info))
    stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logger, logger.error))

    stdout_thread.start()
    stderr_thread.start()

    while True:
        if process.poll() is not None:
            break  # Exit the loop if the process has finished
        else:
            # The process is still running; you can perform additional checks here if needed
            pass
    # Wait for the output threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Wait for the process to complete
    process.wait()

    return process.returncode

class LoggerStream(io.StringIO):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.buffer = ""

    def write(self, buf):
        self.buffer += buf
        lines = self.buffer.split('\n')
        self.buffer = lines.pop()  # Keep the last incomplete line in the buffer

        for line in lines:
            self.process_line(line.rstrip())

    def process_line(self, line):
        # Remove the progress bar
        cleaned_line = re.sub(r'\[=*>\.+\]', '', line).strip()

        # Log lines that don't contain 'ETA'
        if cleaned_line and 'ETA:' not in cleaned_line:
            # Remove leading epoch fraction (e.g., "1/10 - ") if present
            cleaned_line = re.sub(r'^\d+/\d+\s*-\s*', '', cleaned_line)
            if cleaned_line:  # Only log if there's content after cleaning
                self.logger.info(cleaned_line)

    def flush(self):
        if self.buffer:
            self.process_line(self.buffer)
            self.buffer = ""



@contextlib.contextmanager
def redirect_stdout_to_logger(logger):
    logger_stream = LoggerStream(logger)
    old_stdout = sys.stdout
    sys.stdout = logger_stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
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
# CARE Training Functions
##############################################################################


class EpochLoggingCallback(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logger.info(f'Epoch {epoch+1} - loss: {logs.get("loss"):.4f} - val_loss: {logs.get("val_loss"):.4f}')

#Simple care network training with basic augmentation. Can be improved with metrics and additional feedback


#simple augmentation

def add_poisson_noise(image):
    #Add Poisson noise to an image
    scale = 1
    normalized = image / np.max(image)
    noisy = np.random.poisson(normalized * scale) / scale
    return np.clip(noisy, 0, 1) * np.max(image)

def apply_gaussian_blur(image):
    sigma =1
    blurred = gaussian(image, sigma=sigma, preserve_range=True)
    return blurred

def apply_intensity_scale(image):
    intesity_scale = 1.2
    if image.dtype != np.uint16:
        image_16bit = img_as_uint(image)
    else:
        image_16bit = image
    
    adjusted = rescale_intensity(image, out_range=(0, 65535))
        # Scale the intensity based on the factor
    adjusted = np.clip(adjusted * intesity_scale, 0, 65535).astype(np.uint16)
    return adjusted

def rotate_image(img, degrees):
    #Rotate image by degrees.
    rotated_img = np.zeros_like(img)
    # Iterate over each Z slice and rotate it
    for z in range(img.shape[0]):
        rotated_img[z] = rotate(img[z], degrees, reshape=False, mode='reflect')

    return rotated_img

def process_image(image_path, output_dir, degrees, logger, transformation=None):
    #Rotate, optionally add noise/blur, and save the image
    img = imread(image_path)
    #logger.info(f"img.shape: {img.shape}")
    rotated_img = rotate_image(img, degrees)
    #logger.info(f"img.shape: {rotated_img.shape}")

    if transformation:
        if transformation == 'noise':
            rotated_img = add_poisson_noise(rotated_img)
        elif transformation == 'blur':
            rotated_img = apply_gaussian_blur(rotated_img)
        elif transformation == 'intensity':
            rotated_img = apply_intensity_scale(rotated_img)

    os.makedirs(output_dir, exist_ok=True)

    base_name = f"{degrees}_{os.path.splitext(os.path.basename(image_path))[0]}"

    output_path = os.path.join(output_dir, f"{base_name}.tif")
    imwrite(output_path, rotated_img, imagej=True, metadata={'axes': 'ZYX'})
    logger.info(f" Processed and saved: {base_name}")

def augment_images_v1(input_dir, output_dir, logger, is_low_snr=False, choices=None):
    #compatible choices = ['noise', 'blur', 'intensity', None]
    #Process all images with rotation and optional noise/blur
    degrees_list = [90, 180, 270]
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.tif')]
    for degrees in degrees_list:
        for image_path in image_paths:
            transformation = None
            if is_low_snr:
                transformation = random.choice(choices)
            else:
                transformation = None

            process_image(image_path, output_dir, degrees, logger, transformation)



def train_care(inputdir, model_output, model_name, augmentation, image_type, patch_size,
           num_patches_per_img, epochs, unet_kern_size, unet_n_depth,
           batch_size, steps_per_epoch, pct_validation, app, logger):

    model_path = os.path.join(model_output, model_name)
    
    source_dir = 'lowSNR'
    target_dir = 'highSNR'
    
    low_snr_dir = os.path.join(inputdir, source_dir)
    high_snr_dir = os.path.join(inputdir, target_dir)

    #Other Parameters
    #patch_size = (8,128,128)
    #num_patches_per_img = 25
    #epochs = 60
    #unet_kern_size=3 #5 for 2D, else 3
    #unet_n_depth = 3 #default 2
    #batch_size = 16
    #steps_per_epoch = 10
    #pct_validation = 0.10
    initial_learning = 0.0004
    probabilistic = True

            
    if image_type == '2D':
        axes = "YX"
    else:
        axes = "ZYX"
    
    if augmentation == True:    
        
        source_dir = 'augmented_lowSNR'
        target_dir = 'augmented_highSNR'
        
        output_augmented_dir =  os.path.join(inputdir, source_dir)
        output_high_snr_dir =  os.path.join(inputdir, target_dir)
        os.makedirs(output_augmented_dir, exist_ok=True)
        os.makedirs(output_high_snr_dir, exist_ok=True)

        logger.info("Processing Low SNR images with rotation only...")
        augment_images_v1(low_snr_dir, output_augmented_dir, logger, is_low_snr=True, choices =[None] ) #choices = ['noise', 'blur', 'intensity', None]
        
        logger.info("Creating matching high_snr images...")
        augment_images_v1(high_snr_dir, output_high_snr_dir, logger, is_low_snr=False, choices =[None])
        

    #Prepare data
    #logger.info(f"{inputdir}\n{source_dir}\n{target_dir}")

    raw_data = RawData.from_folder (
        basepath    = inputdir,
        source_dirs = [source_dir],
        target_dir  = target_dir,
        axes        = axes,
        pattern='*.tif*'
    )

    #By default, patches are sampled from non-background regions i.e. 
    #that are above a relative threshold that can be given in the function below.

    X, Y, XY_axes = create_patches (
        raw_data            = raw_data,
        patch_size          = patch_size,
        patch_filter        = no_background_patches(threshold=0.4, percentile=99.9),
        n_patches_per_image = num_patches_per_img,
        save_file           = 'CAREdata/my_training_data.npz',
    )
    
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    assert X.shape == Y.shape

    if axes == "YX":
        unet_n_depth = 2 #updating to 2 for 2D - take out of GUI
        logger.info("Training patches information:")
        logger.info(" Shape of X,Y =", X.shape)
        logger.info(" Axes  of X,Y =", XY_axes)

        for i in range(2):
            plt.figure(figsize=(16,4))
            sl = slice(8*i, 8*(i+1)), 0
            plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
            plt.suptitle('Example training pairs');
            # plt.show()
            plt.savefig(model_path + '/Example training pairs.png')
        None;
    if axes == "ZYX":
        logger.info("Training patches information:")
        logger.info(f" Shape of X,Y = { X.shape}")
        logger.info(f" Axes  of X,Y ={ XY_axes}")
        for i in range(2):
            plt.figure(figsize=(16, 4))
            sl = slice(8 * i, 8 * (i + 1)), 0
            plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
            plt.suptitle('Example training pairs (MIPs of 3D patches)');
            #plt.show()
            plt.savefig(model_path + '/Example MIPs of training pairs.png')
        None;


    #Training
    (X,Y), (X_val,Y_val), axes = load_training_data('CAREdata/my_training_data.npz', validation_split=pct_validation, verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    #val figure
    plt.figure(figsize=(12,5))
    plot_some(X_val[:5],Y_val[:5])
    plt.suptitle('5 example validation patches (top row: source, bottom row: target)');
    plt.savefig(model_path+'/5 example validation patches pre training.png')
    
    #temporarily added to address CARE issues with this enviornment - rebuild envioronment to resolve
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    logger.info("\nDisplaying training configuration:")
    logger.info(f"axes: {axes}, n_channel_in: {n_channel_in}, n_channel_out: {n_channel_out}, "
                f"probabilistic: {probabilistic}, steps_per_epoch: {steps_per_epoch}, epochs: {epochs}, "
                f"unet_kern_size: {unet_kern_size}, unet_n_depth: {unet_n_depth}, batch_size: {batch_size}, "
                f"initial_learning: {initial_learning}")
    config = Config(axes, n_channel_in, n_channel_out,
                    probabilistic= probabilistic,
                    train_steps_per_epoch=steps_per_epoch, train_epochs=epochs, unet_kern_size=unet_kern_size,
                    unet_n_depth=unet_n_depth, train_batch_size=batch_size, train_learning_rate=initial_learning)

    logger.info(config)
    vars(config)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info("\nGPU available:")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            gpu_details = tf.config.experimental.get_device_details(gpu)
            logger.info(f"     {gpu_details['device_name']}")


    logger.info("\nTraining CARE network... \n please allow 30min - 3 hrs depending on GPU resources.")

    model = CARE(config, model_name , basedir=model_output)

    start = time.time()
    best_val_loss = float('inf')

   # if app == True:
    with redirect_stdout_to_logger(logger):
        history = model.train(X, Y, validation_data=(X_val, Y_val))

    logger.info(" Training, done.")

    # convert the history.history dict to a pandas DataFrame:
    lossData = pd.DataFrame(history.history)
    
    if os.path.exists(model_path+"/Quality Control"):
      shutil.rmtree(model_path+"/Quality Control")
    
    os.makedirs(model_path+"/Quality Control")
    
    # The training evaluation.csv is saved (overwrites the Files if needed).
    lossDataCSVpath = model_path+'/Quality Control/training_evaluation.csv'
    with open(lossDataCSVpath, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['loss','val_loss', 'learning rate'])
      for i in range(len(history.history['loss'])):
        writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])
    
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    fig = plt.gcf()
    plt.savefig(model_path+'/training_history.png')
    
    # Displaying the time elapsed for training
    dt = time.time() - start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    logger.info(f"\nTime elapsed: {hour} hour(s), {mins} min(s), {round(sec)} sec(s)\n")


    plt.figure(figsize=(20,12))
    _P = model.keras_model.predict(X_val)
    if config.probabilistic:
        _P = _P[...,:(_P.shape[-1]//2)]
    plot_some(X_val[:5],Y_val[:5],_P[:5],pmax=99.5)
    plt.suptitle('5 example validation patches\n'      
                 'top row: input (source),  '          
                 'middle row: target (ground truth),  '
                 'bottom row: predicted from source');

    plt.savefig(model_path+'/Evaluation of validation images unseen.png')

##############################################################################
# nnUnet Training Functions
##############################################################################
def generate_nnunet_json(input_folder, use_ignore=False):
    imagesTr_path = os.path.join(input_folder, 'imagesTr')
    labelsTr_path = os.path.join(input_folder, 'labelsTr')

    # Count the number of training images
    numTraining = len([f for f in os.listdir(labelsTr_path) if f.endswith('.tif')])

    # Get unique image names for channel_names
    image_files = [f[:-9] for f in os.listdir(imagesTr_path) if f.endswith('.tif')]
    unique_image_names = sorted(set(image_files))

    # Determine the number of channels
    num_channels = len([f for f in os.listdir(imagesTr_path) if f.endswith('.tif')]) // len(unique_image_names)

    channel_names = {str(i): f'channel{i+1}' for i in range(num_channels)}

    # Open the first image in labelsTr and count unique intensities
    first_label_image_path = os.path.join(labelsTr_path, os.listdir(labelsTr_path)[0])
    first_label_image = imread(first_label_image_path)
    unique_intensities = sorted(np.unique(first_label_image))

    labels = {'background': 0}
    num_classes = len(unique_intensities) - (2 if use_ignore else 1)
    for i, intensity in enumerate(unique_intensities[1:num_classes + 1], start=1):
        labels[f'Class_{i}'] = int(intensity)

    if use_ignore:
        # If use_ignore is true, assign the last intensity to 'ignore'
        ignore_intensity = int(unique_intensities[-1])
        labels['ignore'] = ignore_intensity

    # Create the JSON structure
    json_data = {
        "channel_names": channel_names,
        "file_ending": ".tif",
        "labels": labels,
        "numTraining": numTraining
    }

    # Write to JSON file
    output_path = os.path.join(input_folder, 'dataset.json')
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    return json_data

def train_nnUNet(raw, preprocessed, results, datasetID, conda_dir, env_dir, nnUNet_env, logger):
    # Set environment variables
    os.environ['nnUNet_raw'] = raw
    os.environ['nnUNet_preprocessed'] = preprocessed
    os.environ['nnUNet_results'] = results

    # Define the commands
    # Construct the command to activate the conda environment
    #env_path = os.path.join(env_dir, nnUNet_env)
    #os.environ['CONDA_ENVS_DIRS'] = env_path + os.pathsep + os.environ.get('CONDA_ENVS_DIRS', '')

    conda_activate_cmd = os.path.join(conda_dir, 'Scripts', 'activate.bat')
    activate_env = f'call {conda_activate_cmd} {nnUNet_env} && set PATH={os.path.join(env_dir, nnUNet_env, "Scripts")};%PATH%'

    # activate_env = fr"{conda_dir}/Scripts/activate.bat {nnUNet_env} && set PATH={env_dir}/{nnUNet_env}\Scripts;%PATH%"
    #activate_env = fr"set PATH={env_dir}/{nnUNet_env}\Scripts;%PATH%"
    #logger.info(f' {conda_dir}/Scripts/activate.bat {nnUNet_env} && set PATH={env_dir}/{nnUNet_env}/Scripts;%PATH%.')
    # Define the nnUNet commands
    command_plan_and_preprocess = f'{activate_env} && nnUNetv2_plan_and_preprocessRESPAN -d {datasetID} --verify_dataset_integrity'
    command_train = f'nnUNetv2_trainRESPAN {datasetID} 3d_fullres all'

    # Combine the activate environment command with the actual nnUNet commands
    final_cmd_plan_and_preprocess = f'{command_plan_and_preprocess}'
    final_cmd_train = f'{activate_env} && {command_train}'

    # Run the commands
    logger.info('Running nnUNetv2_plan_and_preprocess...')

    return_code = run_process_with_logging(final_cmd_plan_and_preprocess, logger)
    #process_plan_and_preprocess = subprocess.Popen(final_cmd_plan_and_preprocess, shell=True)
    #process_plan_and_preprocess.wait()
    logger.info(' Complete.')

    date = time.strftime("%Y_%m_%d_%H_%M")

    logger.info('\nRunning nnUNetv2_train 3d_fullres all. Please 12-24 hours depending on GPU resources...')
    logger.info(f' Training log may not update until training is complete. If you wish to confirm progress'
                f' you may open the log file located in:\n'
                f' {results}\Dataset{datasetID}\\nnUNetTrainer__nnUNetPlans__3d_fullres\\fold_all\\training_log_{date}.txt\n'
                f' This file will update as training progresses, but the file will not refresh (you will have to close then reopen to recheck status).'
                f'\n You may estimate the time remaining by multiplying the epoch time by 1000.')
    #provide current date in this format 2024_7_26_19_26_54
    #date = time.strftime("%Y_%m_%d_%H_%M")


    return_code = run_process_with_logging(final_cmd_train, logger)
    #process_train = subprocess.Popen(final_cmd_train, shell=True)
    #process_train.wait()
    logger.info(' Complete.')



##############################################################################
# SelfNet Training Functions
##############################################################################

