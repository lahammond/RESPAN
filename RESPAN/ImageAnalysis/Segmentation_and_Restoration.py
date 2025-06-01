# -*- coding: utf-8 -*-
"""
UNet and Restoration functions for spine analysis
==========


"""

__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'GPL-3.0 License (see LICENSE)'
__copyright__ = 'Copyright © 2024 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'

import RESPAN.Main.Main as main
import RESPAN.ImageAnalysis.ImageAnalysis as imgan


import os
import numpy as np
import warnings
import re
import shutil
import sys
import contextlib
import time

from pathlib import Path
import subprocess
import threading
from patchify import patchify

from tifffile import imread, imwrite
from skimage import exposure
from skimage.transform import resize

from csbdeep.models import CARE
import json

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

    if settings.image_restore == False and settings.axial_restore == True:
        # restore axial resolution from raw data
        axial_restore_image(locations.input_dir, settings, locations, logger)
        data = locations.restored + '/selfnet/'


    elif settings.image_restore == True and settings.axial_restore == True:
        # restore axial resolution on CARE restored data
        axial_restore_image(data, settings, locations, logger)
        data = locations.restored + '/selfnet/'

    else:
        data = locations.input_dir

    # create nnunet labels
    # #include options here for alternative unets if required
    # logger.info(f"Imnporting from dir {data}")
    log = nnunet_create_labels(data, settings, locations, logger)

    return log



##############################################################################
# Helper Functions
##############################################################################


def run_external_script(script_path, py_path, args):

    cmd = [str(py_path), "-u", str(script_path)]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]

    print("Executing: %s", " ".join(cmd))

    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line.rstrip())
        if proc.wait():
            raise RuntimeError(f"{script_path} failed (exit {proc.returncode})")


def run_external_script_prev(script_path, conda_env, args_dict):
    args_str = ' '.join(f'--{k} "{v}"' for k, v in args_dict.items())
    command = f'conda run -n {conda_env} python "{script_path}" {args_str}'
    process = subprocess.Popen(command, shell=True)
    process.wait()



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


##############################################################################
# SelfNet Restoration
##############################################################################


def axial_restore_image(inputdir, settings, locations, logger):
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Performing SelfNet axial restoration neuron channel...")
    logger.info("   ** Be aware that SelfNet restoration greatly increases file size (up to 10x) **")
    logger.info(
        "   As a result, storage requirements must account for this, and processing time will be increased accordingly.")
    # SelfNet Inference
    # nnunet_env = 'nnunet' # may need to provide conda dir for max compat
    # model path should be the most recent file in saved models checkpoint
    # update the training file to take this file and place it somewhere safe
    # e.g. model_path=input_dir+'checkpoint/saved_models/deblur_net_60_3200.pkl'

    # find most recent pkl file in model_dir and update that to be the model path
    if settings.selfnet_path != None:

        # parser.add_argument('--input_dir', type=str, required=True, help='The input directory')
        # parser.add_argument('--neuron_ch', type=int, default=0, help='the channel for inference')
        # parser.add_argument('--model_path', type=str, required=True, help='The model path')
        # parser.add_argument('--min_v', type=int, default=0, help='The minimum intensity')
        # parser.add_argument('--max_v', type=int, default=65535, help='The maximum intensity')
        # parser.add_argument('--scale', type=float, default=0.21, help='The resolution scaling factor') XY sampling / z step e.g. 75/150 = 0.5
        # parser.add_argument('--z_step', type=int, default=1, help='The final z-resolution')

        args_dict = {
            'input_dir': inputdir,
            'neuron_ch': settings.neuron_channel - 1,
            'model_path': settings.selfnet_path,
            'min_v': 0,
            'max_v': 65535,
            'scale': settings.input_resXY,
            'z_step': settings.input_resZ
        }
        # logger info all args_dict

        logger.info(f"\n   Input xy-resolution: {settings.input_resXY}. Input Z resolution: {settings.input_resZ}")
        logger.info(f"   Final z-resolution: {settings.input_resXY}")

        run_external_script(
            settings.selfnet_inference_script,
            settings.internal_py_path, args_dict)

        # update resolution for remaing calculations
        settings.input_resZ = settings.input_resXY
        logger.info(
            "Restoration complete.")
        # logger.info(settings.input_resZ)

    else:
        logger.info("SelfNet model path not found in settings file, update settings file - skipping axial restoration.")

    logger.info(
        "-----------------------------------------------------------------------------------------------------")


def selfnet_inference(script_path, respan_env, model_dir, min_v, max_v, scale, z_step, settings):
    # nnunet_env = 'nnunet' # may need to provide conda dir for max compat
    # model path should be the most recent file in saved models checkpoint
    # update the training file to take this file and place it somewhere safe
    # e.g. model_path=input_dir+'checkpoint/saved_models/deblur_net_60_3200.pkl'

    # find most recent pkl file in model_dir and update that to be the model path
    args_dict = {
        'input_dir': input,
        'model_path': model_dir,
        'min_v': min_v,
        'max_v': max_v,
        'scale': scale,
        'z_step': z_step
    }
    run_external_script(
        script_path,
        settings.internal_py_path, args_dict)


##############################################################################
# CARE Restoration
##############################################################################

def restore_image(inputdir, settings, locations, logger):
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Restoring images with CARE models...")

    files = [file_i for file_i in os.listdir(inputdir) if file_i.endswith('.tif')]
    files = sorted(files)

    for file in range(len(files)):
        logger.info(f' Restoring image {files[file]} ')

        image = imread(inputdir + files[file])
        logger.info(f"  Raw data has shape {image.shape}")

        image = imgan.check_image_shape(image, logger)

        restored = np.empty((image.shape[0], image.shape[1], image.shape[2], image.shape[3]), dtype=np.uint16)

        for channel in range(image.shape[1]):
            restore_on = getattr(settings, f'c{channel + 1}_restore', None)
            rest_model_path = getattr(settings, f'c{channel + 1}_rest_model_path', None)
            rest_type = getattr(settings, f'c{channel + 1}_rest_type', None)
            if restore_on == True and rest_model_path != None:
                logger.info(f"  Restoring channel {channel + 1}")
                logger.info(f"  Restoration model = {rest_model_path}\n  ---")

                channel_image = image[:, channel, :, :]

                # find min int above zero and use this to replace all zero values
                min_intensity = np.min(channel_image[np.nonzero(channel_image)])
                max_intensity = np.max(channel_image)
                logger.info(f"  Pre restoration min intensity: {min_intensity}, max intensity: {max_intensity}")

                channel_image = np.where(channel_image == 0, min_intensity, channel_image)

                if rest_type[0] == 'care':
                    if os.path.isdir(rest_model_path) is False:
                        raise RuntimeError(rest_model_path, "not found, check settings and model directory")
                    rest_model = CARE(config=None, name=rest_model_path)

                    # restored = np.empty((channel_image.shape[0], channel_image.shape[1], channel_image.shape[2]), dtype=np.uint16)

                    with suppress_all_output(), main.HiddenPrints():

                        # restore image
                        restored_channel = rest_model.predict(channel_image, axes='ZYX',
                                                              n_tiles=settings.tiles_for_prediction)

                        # convert to 16bit
                        restored_channel = restored_channel.astype(np.uint16)
                        #set intensities 10% above original max ints to min_intensity
                        restored_channel = np.where(restored_channel > max_intensity * 1.2, min_intensity, restored_channel)


                        restored[:, channel, :, :] = restored_channel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if settings.validation_format == "tif":
                imwrite(locations.restored + files[file], restored, compression=('zlib', 1), imagej=True,
                        photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZCYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))

    logger.info(
        "Restoration complete.\n\n-----------------------------------------------------------------------------------------------------")



##############################################################################
# nnUNet Segmentation
##############################################################################


def initialize_nnUnet(settings, logger):
    #not worrying about setting raw and processed, as not needed and would rquire additional params for user/settings file
    #os.environ['nnUNet_raw'] = settings.nnUnet_raw
    #os.environ['nnUNet_preprocessed'] = settings.nnUnet_preprocessed
    nnUnet_results = Path(settings.neuron_seg_model_path).parent
    nnUnet_results = str(nnUnet_results).replace("\\", "/")
    os.environ['nnUNet_results'] = nnUnet_results



def nnunet_create_labels(inputdir, settings, locations, logger):
    time_initial = time.time()
    logger.info(f" Detecting spines and dendrites...")
    settings.shape_error = False
    settings.rescale_req = False

    # check if rescaling required and create scaling factors
    if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
        logger.info(f"  Images will be rescaled to match network.")

        settings.rescale_req = True
        # z in / z desired, y in / desired ...
        settings.scaling_factors = (settings.input_resZ / settings.model_resZ,
                                    settings.input_resXY / settings.model_resXY,
                                    settings.input_resXY / settings.model_resXY)
        # settings.inverse_scaling_factors = tuple(1/np.array(settings.scaling_factors))

        logger.info(
            f"  Scaling factors: Z = {round(settings.scaling_factors[0], 2)} Y = {round(settings.scaling_factors[1], 2)} X = {round(settings.scaling_factors[2], 2)} ")

    # data can be raw data OR restored data so check channels

    files = [file_i
             for file_i in os.listdir(locations.input_dir)
             if file_i.endswith('.tif')]
    files = sorted(files)

    label_files = [file_i
                   for file_i in os.listdir(locations.labels)
                   if file_i.endswith('.tif')]

    # create empty arrays to capture dims and padding info
    settings.original_shape = [None] * len(files)
    settings.padding_req = np.zeros(len(files))

    if len(files) == len(label_files):
        logger.info(
            f"  *Spines and dendrites already detected. \nDelete \Validation_Data\Segmentation_Labels if you wish to regenerate.")
        stdout = None
        settings.prev_labels = True

        return_code = 0
    else:

        # Prepare Raw data for nnUnet

        # Initialize reference to None - if using histogram matching
        # logger.info(f" Histogram Matching is set to = {settings.HistMatch}")
        reference_image = None
        settings.prev_labels = False

        for file in range(len(files)):
            logger.info(f"   Preparing image {file + 1} of {len(files)} - {files[file]}")

            image = imread(inputdir + files[file])
            logger.info(f"   Raw data has shape: {image.shape}")

            image = imgan.check_image_shape(image, logger)

            if settings.axial_restore == True:
                neuron = image[:, 0, :, :]
            else:
                neuron = image[:, settings.neuron_channel - 1, :, :]

            # rescale if required by model
            if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                settings.original_shape[file] = neuron.shape
                # new_shape = (int(neuron.shape[0] * settings.scaling_factors[0]), neuron.shape[1] * settings.scaling_factors[1]), neuron.shape[2] * settings.scaling_factors[2]))
                new_shape = tuple(int(dim * factor) for dim, factor in zip(neuron.shape, settings.scaling_factors))
                neuron = resize(neuron, new_shape, mode='constant', preserve_range=True, anti_aliasing=True)
                logger.info(f"   Data rescaled to match model for labeling has shape: {neuron.shape}")

            # logger.info the  it will take for processing image by dividing the number of pixels by a scaling factor
            # limit variable to 2 decimal places

            logger.info(
                f"   Estimated time to detect spines and dendrites for this image: {round(neuron.size * 2.5e-8, 2)} minutes.\n")
            if neuron.shape[0] < 5:
                # settings.shape_error = True
                # logger.info(f"  !! Insufficient Z slices - please ensure 5 or more slices before processing.")
                # logger.info(f"  File has been moved to \\Not_processed and excluded from processing.")
                # if not os.path.isdir(inputdir+"Not_processed/"):
                #    os.makedirs(inputdir+"Not_processed/")
                # os.rename(inputdir+files[file], inputdir+"Not_processed/"+files[file])

                # Padding and flag this file as padded for unpadding later
                # Pad the array
                settings.padding_req[file] = 1
                neuron = np.pad(neuron, pad_width=((2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
                logger.info(f"   Too few Z-slices, padding to allow analysis.")

            if settings.HistMatch == True:
                # If reference_image is None, it's the first image.
                if reference_image is None:
                    # neuron = contrast_stretch(neuron, pmin=0, pmax=100)
                    reference_image = neuron
                else:
                    # Match histogram of current image to the reference image
                    neuron = exposure.match_histograms(neuron, reference_image)

            # logger.info(f" ")
            # save neuron as a tif file in nnUnet_input - if file doesn't end with 0000 add that at the end
            name, ext = os.path.splitext(files[file])

            if not name.endswith("0000"):
                name += "_0000"

            new_filename = name + ext

            filepath = locations.nnUnet_input + new_filename

            byte_limit = int(3.5 * 1024 ** 3)  # 3.5 GB threshold
            need_bigtiff = neuron.nbytes > byte_limit  # uint16 → 2 bytes/voxel

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imwrite(
                    filepath,
                    neuron.astype(np.uint16),
                    compression=('zlib', 1),
                    photometric='minisblack',
                    metadata={'spacing': settings.input_resZ,
                              'unit': 'um',
                              'axes': 'ZYX'},
                    resolution=(settings.input_resXY, settings.input_resXY),
                    imagej=not need_bigtiff,
                    bigtiff=need_bigtiff,
                )

        # Run nnUnet over prepared files
        # initialize_nnUnet(settings)

        if settings.patch_for_nnunet:  # True ⇢ use tiling workflow
            logger.info("  Creating nnUNet input tiles …")
            patch_images_for_nnunet(locations.nnUnet_input, settings, logger)


        # split the path into subdirectories
        subdirectories = os.path.normpath(settings.neuron_seg_model_path).split(os.sep)
        last_subdirectory = subdirectories[-1]
        # find all three digit sequences in the last subdirectory
        matches = re.findall(r'\d{3}', last_subdirectory)
        # If there's a match, assign it to a variable
        dataset_id = matches[0] if matches else None

        logger.info("  Performing spine and dendrite detection on GPU...")

        ##uncomment if issues with nnUnet
        # logger.info(f"{settings.nnUnet_conda_path} , {settings.nnUnet_env} , {locations.nnUnet_input}, {locations.labels} , {dataset_id} , {settings.nnUnet_type} , {settings}")

        return_code = run_nnunet_predict(settings.nnunet_predict_bat,
                                         locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type,
                                         settings, logger)

        if settings.patch_for_nnunet:
            logger.info("  Re-assembling nnUNet tile outputs …")
            reassemble_patch_predictions(locations.labels,
                                         locations.nnUnet_input,
                                         locations.labels,
                                         settings, logger)
        # logger.info(cmd)

        ##uncomment if issues with nnUnet
        # result = run_nnunet_predict(settings.nnUnet_conda_path, settings.nnUnet_env, locations.nnUnet_input, locations.labels, dataset_id, settings.nnUnet_type, locations,settings)

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
        # logger.info(stdout)

        # delete nnunet input folder and files
        if settings.save_intermediate_data == False and settings.Track == False:
            if os.path.exists(locations.nnUnet_input):
                shutil.rmtree(locations.nnUnet_input)

        # Clean up label folder

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

        # if tracking over time then we want unpad and match how the labels will appear

        files = [file_i
                 for file_i in os.listdir(locations.labels)
                 if file_i.endswith('.tif')]
        files = sorted(files)

        for file in range(len(files)):
            # if file == 0: logger.info(' Unpadding and rescaling neuron channel for registration and time tracking...')

            # Unpad if padded # later update - these can be included in Unet processing stage to simplify!
            if settings.padding_req[
                file] == 1:  # and settings.prev_labels == False and settings.original_shape[0] != None:
                logger.info(f"  Unpadding image {files[file]}")
                image = imread(locations.labels + files[file])
                image = image[2:-2, :, :]

                logger.info(f"  Image {files[file]} has shape: {image.shape}")

                imwrite(locations.labels + files[file], image.astype(np.uint8), compression=('zlib', 1), imagej=True,
                        photometric='minisblack',
                        metadata={'spacing': settings.input_resZ, 'unit': 'um', 'axes': 'ZYX', 'mode': 'composite'},
                        resolution=(settings.input_resXY, settings.input_resXY))

                logger.info(f"  Image {files[file]} has been unpaded and saved to {locations.labels}")

                # Unpad if padded
                # if settings.padding_req[file] == 1:
                #    image = image[2:-2, :, :]
                # image = imread(locations.nnUnet_input + files[file])
                # logger.info(f"  Image {files[file]} has shape: {image.shape}")

                # rescale labels back up if required
                # if settings.input_resZ != settings.model_resZ or settings.input_resXY != settings.model_resXY:
                # logger.info(f"orignal settings shape: {settings.original_shape[file]}")
                #    image = resize(image, settings.original_shape[file], order=0, mode='constant', preserve_range=True, anti_aliasing=None)
                # logger.info(f"image resized: {labels.shape}")

                #    imwrite(locations.nnUnet_input + files[file], image.astype(np.uint8), imagej=True, photometric='minisblack',
                #           metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                #          resolution=(settings.input_resXY, settings.input_resXY))
        total_time = (time.time() - time_initial) / 60
        logger.info(
            f"\n  Total time for spine and dendrite label creation: {round(total_time, 2)} minutes.")
    # logger.info("Segmentation complete.\n")
    logger.info(
        "\n-----------------------------------------------------------------------------------------------------")
    return return_code


def run_nnunet_predict(nnunet_predict_bat, input_dir, output_dir, dataset_id, nnunet_type, settings, logger):
    # Set environment variables

    initialize_nnUnet(settings, logger)

    #activate_env = fr"{conda_dir}\Scripts\activate.bat {nnUnet_env}&& set PATH={settings.nnUnet_env_path}/{nnUnet_env}/Scripts;%PATH%"

    # Define the command to be run
    # cmd = "nnUNetv2_predictRESPAN -i \"{}\" -o \"{}\" -d {} -c {} -f all".format(input_dir, output_dir, dataset_id, nnunet_type)
    #cmd = "nnUNetv2_predict -i \"{}\" -o \"{}\" -d {} -c {} -f all".format(input_dir, output_dir, dataset_id,nnunet_type)

    cmd = [
        str(nnunet_predict_bat),
        "-i", input_dir,
        "-o", output_dir,
        "-d", dataset_id,
        "-c", nnunet_type,
        "-f", "all"
    ]

    # Combine the activate environment command with the actual command
    #final_cmd = f'{activate_env} && {cmd}'

    return_code, stdout_out, stderr_out = run_process_with_logging(cmd, logger)
    # Run the command
    if return_code != 0:
        logger.info(f"Error: Command failed with return code {return_code}")
        if stderr_out:
            logger.info(f"Error message:\n{stderr_out}")
        if stdout_out:
            logger.info(f"Output message:\n{stdout_out}")

    # process = subprocess.Popen(final_cmd, shell=True)
    # stdout, stderr = process.communicate()
    return return_code


def _mapping_path(nn_input_dir):
    return os.path.join(nn_input_dir, "_patch_map.json")


def patch_images_for_nnunet(nn_input_dir, settings, logger):
    """
    Split every *.tif in nn_input_dir into overlapping blocks, save them
    as *_patchXXXX_0000.tif, and drop the original file.  A JSON map is
    written so we can stitch the predictions later.
    """

    mapping = []

    for fn in [f for f in os.listdir(nn_input_dir) if f.lower().endswith(".tif")]:
        img_path = os.path.join(nn_input_dir, fn)
        img = imread(img_path)  # (Z,Y,X)

        # ---- per-image patch / stride that always fit --------------------
        psize = tuple(min(p, d) for p, d in zip(settings.nnunet_patch_size, img.shape))
        stride = tuple(min(s, p) for s, p in zip(settings.nnunet_stride, psize))

        #  patchify never fails now
        patches = patchify(img, psize, step=stride)  # shape: (nZ,nY,nX,*psize)
        nZ, nY, nX = patches.shape[:3]
        base, idx = fn[:-4], 0

        logger.info(f"    Patching {fn}  →  {nZ * nY * nX} blocks  (patch {psize}, stride {stride})")
        for z in range(nZ):
            for y in range(nY):
                for x in range(nX):
                    imwrite(os.path.join(
                        nn_input_dir,
                        f"{base}_patch{idx:04d}_0000.tif"),
                        patches[z, y, x].astype(np.uint16))
                    idx += 1

        mapping.append(
            {"base": base,
             "shape": img.shape,
             "nZ": nZ, "nY": nY, "nX": nX,
             "patch_size": psize,
             "stride": stride}
        )
        os.remove(img_path)  # avoid double processing

    with open(_mapping_path(nn_input_dir), "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping

def reassemble_patch_predictions(nn_output_dir, nn_input_dir,
                                 final_dir, settings, logger):
    """
    Stitch nnUNet tile predictions by majority vote.
    Class count determined on-the-fly.
    """
    with open(_mapping_path(nn_input_dir)) as f:
        mapping = json.load(f)

    for entry in mapping:
        base = entry["base"]
        base_out = base[:-5] if base.endswith("_0000") else base
        shape = tuple(entry["shape"])
        nZ, nY, nX = entry["nZ"], entry["nY"], entry["nX"]
        psize = tuple(entry["patch_size"])
        stride = tuple(entry["stride"])

        # pass 1 ─ find max label
        max_lab, idx = 0, 0
        for _ in range(nZ * nY * nX):
            p = os.path.join(nn_output_dir, f"{base}_patch{idx:04d}.tif")
            if os.path.exists(p):
                m = imread(p).max()
                if m > max_lab:
                    max_lab = m
            idx += 1

        n_classes = int(max_lab) + 1
        counts = np.zeros((n_classes, *shape), dtype=np.uint16)

        # pass 2 ─ vote accumulation
        idx = 0
        for z in range(nZ):
            z0 = z * stride[0]
            for y in range(nY):
                y0 = y * stride[1]
                for x in range(nX):
                    x0 = x * stride[2]
                    p = os.path.join(nn_output_dir, f"{base}_patch{idx:04d}.tif")
                    if os.path.exists(p):
                        patch = imread(p)
                        for c in range(n_classes):
                            mask = patch == c
                            if mask.any():
                                counts[c,
                                z0:z0 + psize[0],
                                y0:y0 + psize[1],
                                x0:x0 + psize[2]] += mask
                    idx += 1

        seg = np.argmax(counts, axis=0).astype(np.uint16)
        out_path = os.path.join(final_dir, f"{base_out}.tif")
        imwrite(out_path, seg, compression="zlib")
        logger.info(f"    Re-assembled: {base_out}.tif")
        for f in os.listdir(nn_output_dir):
            if f.endswith(".tif") and "_patch" in f:
                os.remove(os.path.join(nn_output_dir, f))

