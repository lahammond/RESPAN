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


# Code to register timelapse images and then track spines

import os
import numpy as np 

import ants

import warnings

import nibabel as nib
from tifffile import imread, imwrite

import matplotlib.pyplot as plt

from pathlib import Path

from skimage import img_as_uint
from skimage import exposure
from skimage.measure import label
from skimage.measure import regionprops
from scipy.spatial import distance


"""    

tiff_to_nifti(inputdir, inputdir+"/image_nii")
labels_to_nifti(inputdir+"/Validation_Data/Segmentation_Labels", inputdir+"/labels_nii")

#registered_images = serial_registration_from_folder_rigid(inputdir+"/nii")
registered_images, registered_labels = serial_registration_of_images_and_labels(inputdir+"/image_nii", inputdir+"/labels_nii")

save_registered_images_to_tif(registered_images, inputdir+"/registered")
save_registered_images_to_tif(registered_labels, inputdir+"/registered_labels")

final_array = save_as_4d_tiff(registered_images, inputdir+"/registered/", "4d.tif")
final_array = save_as_4d_tiff(registered_labels, inputdir+"/registered_labels/", "4d.tif")

"""

def track_spines(settings, locations, log, logger):
    logger.info("Registering and tracking spines...")
    
    #enhance nnUnet for registration if necessary
    #enhance_for_reg(locations.nnUnet_input, locations.nnUnet_input+"/enhanced/", settings, logger)

    #Convert to nifti for reg
    #tiff_to_nifti(locations.nnUnet_input+"/enhanced/", locations.nnUnet_input+"/enhanced_image_nii/", logger)
    #tiff_to_nifti(locations.nnUnet_input, locations.nnUnet_input+"/image_nii/", logger)
    #labels_to_nifti(locations.labels, locations.labels+"/labels_nii/", logger)

    #registration - modify to include translation of all channels in raw data    
    #registered_images, registered_labels = serial_registration_of_images_and_labels(locations.nnUnet_input+"/enhanced_image_nii",locations.nnUnet_input+"/image_nii", locations.labels+"/labels_nii", settings, logger)
    
    #needs to be modified to ensure registers neuron channel and translates other channels
    registered_images, registered_labels = serial_registration_of_np_images_and_labels(settings, locations, logger)


    #save_registered_images_to_tif(registered_images, locations.nnUnet_input+"/registered", logger)
    #save_registered_images_to_tif(registered_labels, locations.input_dir+"/Validation_Data/Registered_segmentation_labels", logger)

    final_array = save_as_4d_tiff(registered_images, locations.input_dir+"/Registered/", "Registered_images_4d.tif", "Registered_images_4d_MIP.tif", logger)
    final_array = save_as_4d_tiff(registered_labels, locations.input_dir+"/Registered/", "Registered_labels_4d.tif", "Registered_labels_4d_MIP.tif", logger)
    
   

def enhance_for_reg(inputdir, outputdir, settings, logger):
    #some code here for 
    # file in files
    logger.info("Enhancing for registration...")
    
    #Create output
    if not os.path.exists(outputdir):
       os.makedirs(outputdir)
    
    files = [file_i
             for file_i in os.listdir(inputdir)
             if file_i.endswith('.tif')]
    files = sorted(files)
    
    for file in range(len(files)):
        image = imread(inputdir + files[file])
        image = contrast_stretch(image, pmin=2, pmax=97)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imwrite(outputdir + files[file], image.astype(np.uint16), imagej=True, photometric='minisblack',
                    metadata={'spacing': settings.input_resZ, 'unit': 'um','axes': 'ZYX', 'mode': 'composite'},
                    resolution=(settings.input_resXY, settings.input_resXY))

    
    """
    if settings.HistMatch == True:
        # If reference_image is None, it's the first image.
        if reference_image is None:
            neuron = contrast_stretch(neuron, pmin=0, pmax=100)
            reference_image = neuron
        else:
           # Match histogram of current image to the reference image
           neuron = exposure.match_histograms(neuron, reference_image) 
           """

def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))



def save_as_4d_tiff(images, outputdir, filename, filename2, logger):
    """
    Save a 4D numpy array as a TIFF image.
    
    Parameters:
        array (np.array): 4D numpy array of image data.
        outputpath (str): Path where the 4D TIFF image will be saved.
    """
    #logger.info("Saving registered 4D data...")
    #Create output
    if not os.path.exists(outputdir):
       os.makedirs(outputdir)
       
    # Convert each ANTs image to a numpy array and stack to form a 4D array
    #stacked_array = np.stack([img.numpy() for img in images], axis=0)
    #logger.info(stacked_array.shape)
    reordered_array = np.moveaxis(images, [0, 1, 2, 3], [0, 3, 2, 1])
    #add an axis for C at position 0
    final_array = reordered_array[:, :, np.newaxis,  :, :]
    
    #logger.info(final_array.shape)
    
    #final_array = img_as_uint(final_array)
    
    imwrite(outputdir+filename, final_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': 1, 'unit': 'um','axes': 'TZCYX', 'mode': 'composite'})
    
    final_array = np.max(final_array, axis=1)
    imwrite(outputdir+filename2, final_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': 1, 'unit': 'um','axes': 'TCYX', 'mode': 'composite'})
    
    return(final_array)


def serial_registration_of_np_images_and_labels(settings, locations, logger):
    """
    Serially register a list of images using rigid body registration with ANTsPy.
    
    Parameters:
        images_list (list of str): List of image paths in order they should be registered.
    
    Returns:
        list of ANTsImage: List of registered images.
    """
    
    # Sort filenames to ensure the correct order
    #enhanced_list = sorted([os.path.join(enhanced_input_path, filename) for filename in os.listdir(enhanced_input_path) if filename.endswith('.nii')])
    
    images_list = sorted([os.path.join(locations.nnUnet_input, filename) for filename in os.listdir(locations.nnUnet_input) if filename.endswith('.tif')])

    labels_list = sorted([os.path.join(locations.labels, filename) for filename in os.listdir(locations.labels) if filename.endswith('.tif')])

    if not images_list:
        raise ValueError("No images found in the provided folder.")

    

    #values for direct converstion from np to ants
    ants_spacing =(1.0, 1.0, 1.0)
    ants_origin = (0.0, 0.0, 0.0)
    ants_direction = ([[-1.,  0.,  0.], [ 0., -1.,  0.], [ 0.,  0.,  1.]])
   

    #registered_enhanced = []
    registered_images = []
    registered_labels = []
    
    
    #read in Tifs
    # The first image is considered the fixed image for the first registration
    #fixed_enhanced = imread(enhanced_list[0])
    fixed_image = imread(images_list[0])
    fixed_label = imread(labels_list[0])
    
    
    #Perform any modifications on Tifs
    #fixed_enhanced[fixed_enhanced == 0] = 1
    #pad the arrays with black for better resutls - if using elastic
    pad_width = 10 #10 - put these parameters in settings file if necessary
    minSlices = 1 #5 - disabled but use for z rescale if necessary
    rescale = 0.3 # - z rescale if minslices enabled
    
    #swap axes for ants and pad if necessary
    fixed_image = np.swapaxes(np.pad(fixed_image, pad_width), 0, 2)
    fixed_label = np.swapaxes(np.pad(fixed_label, pad_width), 0, 2)
       

    
    #rescale if necessary
    if fixed_image.shape[2] < minSlices:
        #fixed_enhanced = ants.resample_image(fixed_enhanced, (1, 1, rescale), use_voxels=False, interp_type=4)
        fixed_image = ants.resample_image(fixed_image, (1, 1, rescale), use_voxels=False, interp_type=4)
        fixed_label = ants.resample_image(fixed_label, (1, 1, rescale), use_voxels=False, interp_type=0)
    
    #convert to ants
    #fixed_enhanced = ants.from_numpy(fixed_enhanced.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)
    fixed_image = ants.from_numpy(fixed_image.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)
    fixed_label = ants.from_numpy(fixed_label.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)

    #append to image list
    #registered_enhanced.append(fixed_enhanced)
    registered_images.append(fixed_image)
    registered_labels.append(fixed_label)
    
    
    logger.info("Registering and transforming images...")

    for i in range(1, len(images_list)):
        logger.info(f" Registering images {i} and {i+1} of {len(images_list)}.")
        #moving_enhanced = np.swapaxes(imread(enhanced_list[i]), 0, 2)
        moving_image = np.swapaxes(imread(images_list[i]), 0, 2)
        moving_label = np.swapaxes(imread(labels_list[i]), 0, 2)
        
        #Pad
        #moving_enhanced = np.pad(moving_enhanced, pad_width)
        moving_image = np.pad(moving_image, pad_width)
        moving_label = np.pad(moving_label, pad_width)  
        
        
        #mask testing
        #mask2 = moving_label > 0
        #moving_image = np.where(mask2, moving_image, 0)
        
        #rescale if necessary
        if moving_image.shape[2] < minSlices:
            #moving_enhanced = ants.resample_image(moving_enhanced, (1, 1, rescale), use_voxels=False, interp_type=4)
            moving_image = ants.resample_image(moving_image, (1, 1, rescale), use_voxels=False, interp_type=4)
            moving_label = ants.resample_image(moving_label, (1, 1, rescale), use_voxels=False, interp_type=0)
        
        #Convert to ants
        #convert to ants
        #moving_enhanced = ants.from_numpy(moving_enhanced.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)
        moving_image = ants.from_numpy(moving_image.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)
        moving_label = ants.from_numpy(moving_label.astype("float32"), origin=ants_origin, spacing=ants_spacing, direction=ants_direction)
        
        #print("Rigid reg")
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='DenseRigid', #Rigid
            initial_transform=None, 
            outprefix='', 
            mask=None, 
            moving_mask=None, 
            mask_all_stages=False, 
            grad_step=0.1,  #0.2
            flow_sigma=0, #3 
            total_sigma=0, 
            aff_metric='mattes', 
            aff_sampling=20, #32
            aff_random_sampling_rate=0.2, #0.2
            syn_metric='mattes', 
            syn_sampling=10, #32
            reg_iterations=(200, 100, 50),   #(40, 20, 0) #20000, 10000, 5000
            write_composite_transform=False, 
            random_seed=None, 
            verbose=False, 
            multivariate_extras=None, 
            restrict_transformation=None, 
            smoothing_in_mm=False

        )
        
        
        if settings.reg_method == "Elastic":
            print("Elastic reg")
            # Stage 2: Affine Registration (optional, can be included for additional refinement)
            elastic_registration = ants.registration(
                fixed=fixed_image,
                moving=registration['warpedmovout'],
                type_of_transform='SyNAggro',
                mask_all_stages=False
                #grad_step=0.8, #gradient step size (not for all tx)
                #flow_sigma=3, #smoothing for update field
                #total_sigma=0, #smoothing for total field
                #aff_metric='mattes', #the metric for the affine part (GC, mattes, meansquares)
                #aff_sampling=64, #the nbins or radius parameter for the syn metric
                #aff_random_sampling_rate=0.8, #the fraction of points used to estimate the metric. this can impact speed but also reproducibility and/or accuracy.
                #syn_metric='mattes', #the metric for the syn part (CC, mattes, meansquares, demons)
                #syn_sampling=32, #the nbins or radius parameter for the syn metric
                #reg_iterations=(2000, 2000, 50), # vector of iterations for syn. we will set the smoothing and multi-resolution parameters based on the length of this vector.
                #aff_iterations=(100, 100, 50), #vector of iterations for low-dimensional (translation, rigid, affine) registration.
                #aff_shrink_factors=(0.5, 0, 0), #vector of multi-resolution shrink factors for low-dimensional (translation, rigid, affine) registration.
                #aff_smoothing_sigmas=(0.5, 0, 0), #vector of multi-resolution smoothing factors for low-dimensional (translation, rigid, affine) registration.
                #write_composite_transform=False, #Boolean specifying whether or not the composite transform (and its inverse, if it exists) should be written to an hdf5 composite file. This is false by default so that only the transform for each stage is written to file.
                #random_seed=None, #random seed to improve reproducibility. note that the number of ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS should be 1 if you want perfect reproducibility.
                #verbose=False, #request verbose output (useful for debugging)
                #multivariate_extras=None, #list of additional images and metrics which will trigger the use of multiple metrics in the registration process in the deformable stage. Each multivariate metric needs 5 entries: name of metric, fixed, moving, weight, samplingParam. the list of lists should be of the form ( ( “nameOfMetric2”, img, img, weight, metricParam ) ). Another example would be ( ( “MeanSquares”, f2, m2, 0.5, 0
                #, ( “CC”, f2, m2, 0.5, 2 ) ) . This is only compatible    with the SyNOnly or antsRegistrationSyN* transformations.
                #restrict_transformation=None, #This option allows the user to restrict the) – optimization of the displacement field, translation, rigid or affine transform on a per-component basis. For example, if one wants to limit the deformation or rotation of 3-D volume to the first two dimensions, this is possible by specifying a weight vector of ‘(1,1,0)’ for a 3D deformation field or ‘(1,1,0,1,1,0)’ for a rigid transformation. Restriction currently only works if there are no preceding transformations.
                #smoothing_in_mm=False #(boolean ; currently only impacts low dimensional registration)
      
      
                ) 
          
          
            #combine transforms
            fwd_transform = registration['fwdtransforms'] + \
                            elastic_registration['fwdtransforms']
          
      
        #warped_label = ants.apply_transforms(fixed=fixed_enhanced, moving=moving_label, transformlist=registration['fwdtransforms'], interpolator='genericLabel') #'nearestNeighbor')
        #warped_image = ants.apply_transforms(fixed=fixed_enhanced, moving=moving_image, transformlist=registration['fwdtransforms'], interpolator='linear')
        if settings.reg_method == "Rigid":
            warped_label = ants.apply_transforms(fixed=fixed_image, moving=moving_label, transformlist=registration['fwdtransforms'], interpolator='genericLabel') #'nearestNeighbor')
            warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration['fwdtransforms'], interpolator='linear')
            # The registered image becomes the new fixed image for the next iteration
            fixed_image = registration['warpedmovout'] 
      
        if settings.reg_method == "Elastic":
            warped_label = ants.apply_transforms(fixed=fixed_image, moving=moving_label, transformlist=fwd_transform, interpolator='genericLabel', defaultvalue=0) #'nearestNeighbor')
            warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=fwd_transform, interpolator='linear', defaultvalue=0)
            fixed_image = elastic_registration['warpedmovout']
              
        
        registered_images.append(warped_image)
        registered_labels.append(warped_label)
       
    if pad_width >0:
        for i in range(len(registered_images)):
            registered_images[i] = registered_images[i][pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]
            registered_labels[i] = registered_labels[i][pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

    # Stack along a new first axis to create 4D array
    registered_images = np.stack(registered_images, axis=0)
    registered_labels = np.stack(registered_labels, axis=0)
    
    logger.info(f"Registration complete. \n")

    return registered_images, registered_labels

def serial_registration_from_folder_rigid(input_path, logger):
    """
    Serially register a list of images using rigid body registration with ANTsPy.
    
    Parameters:
        images_list (list of str): List of image paths in order they should be registered.
    
    Returns:
        list of ANTsImage: List of registered images.
    """

    
    # Sort filenames to ensure the correct order
    images_list = sorted([os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.endswith('.nii')])

    if not images_list:
        raise ValueError("No images found in the provided folder.")

    
    registered_images = []
    # The first image is considered the fixed image for the first registration
    fixed_image = ants.image_read(images_list[0])
    registered_images.append(fixed_image)
    logger.info(f"Registering and transforming images and labels...")
    for i in range(1, len(images_list)):
        #logger.info(f"Processing: {images_list[i]}")
        moving_image = ants.image_read(images_list[i])
        
        # Register moving image to fixed image using rigid transformation
        #registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN') - non-linear symmetric normalization transformation
        #registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
        
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='DenseRigid', #Rigid
            initial_transform=None, 
            outprefix='', 
            mask=None, 
            moving_mask=None, 
            mask_all_stages=False, 
            grad_step=0.1,  #0.2
            flow_sigma=0, #3 
            total_sigma=0, 
            aff_metric='mattes', 
            aff_sampling=20, #32
            aff_random_sampling_rate=0.5, #0.2
            syn_metric='mattes', 
            syn_sampling=10, #32
            reg_iterations=(20000, 20000, 20000),   #(40, 20, 0)
            #aff_iterations=(2100, 1200, 1200, 10), 
            #aff_shrink_factors=(0, 0, 0, 0),  #(6, 4, 2, 1),
            #aff_smoothing_sigmas=(0, 0, 0, 0),  #(3, 2, 1, 0)
            write_composite_transform=False, 
            random_seed=None, 
            verbose=False, 
            multivariate_extras=None, 
            restrict_transformation=None, 
            smoothing_in_mm=False
            #grad_step=0.05,  # gradient step, you can adjust based on your images
            #metric_weight=1,
            #radius_or_number_of_bins=100, # e.g., use for MI
            #sampling_strategy='Regular',
            #sampling_percentage=0.80,
            #number_of_iterations=[300],
            #convergence_threshold=1e-6,
            #convergence_window_size=100,
            #shrink_factors=[[0, 0, 0]],
            #smoothing_sigmas=[[0, 0, 0]]
        )
        
        
        # The registered image becomes the new fixed image for the next iteration
        fixed_image = registration['warpedmovout']
        registered_images.append(fixed_image)
    
    # Stack along a new first axis to create 4D array
    registered_images = np.stack(registered_images, axis=0)

    logger.info(f" Registration complete.\n")
    
    return registered_images


def serial_registration_of_images_and_labels(enhanced_input_path, img_input_path, labels_input_path, settings, logger):
    """
    Serially register a list of images using rigid body registration with ANTsPy.
    
    Parameters:
        images_list (list of str): List of image paths in order they should be registered.
    
    Returns:
        list of ANTsImage: List of registered images.
    """
    
    # Sort filenames to ensure the correct order
    enhanced_list = sorted([os.path.join(enhanced_input_path, filename) for filename in os.listdir(enhanced_input_path) if filename.endswith('.nii')])
    
    images_list = sorted([os.path.join(img_input_path, filename) for filename in os.listdir(img_input_path) if filename.endswith('.nii')])

    labels_list = sorted([os.path.join(labels_input_path, filename) for filename in os.listdir(labels_input_path) if filename.endswith('.nii')])

    if not images_list:
        raise ValueError("No images found in the provided folder.")

    
    #registered_enhanced = []
    registered_images = []
    registered_labels = []

    
    # The first image is considered the fixed image for the first registration
    fixed_enhanced = ants.image_read(enhanced_list[0])
    fixed_image = ants.image_read(images_list[0])
    fixed_label = ants.image_read(labels_list[0])
       
    #registered_enhanced.append(fixed_enhanced)
    registered_images.append(fixed_image)
    registered_labels.append(fixed_label)
    
    logger.info("Registering and transforming images...")

    for i in range(1, len(images_list)):
        logger.info(f" Processing {i} of {len(images_list)}.")
        moving_enhanced = ants.image_read(enhanced_list[i])
        
        moving_image = ants.image_read(images_list[i])

        moving_label = ants.image_read(labels_list[i])
        
        # Register moving image to fixed image using rigid transformation
        #registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN') - non-linear symmetric normalization transformation
        #registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
        
        registration = ants.registration(
            fixed=fixed_enhanced,
            moving=moving_enhanced,
            type_of_transform='DenseRigid', #Rigid
            initial_transform=None, 
            outprefix='', 
            mask=None, 
            moving_mask=None, 
            mask_all_stages=False, 
            grad_step=0.1,  #0.2
            flow_sigma=0, #3 
            total_sigma=0, 
            aff_metric='mattes', 
            aff_sampling=20, #32
            aff_random_sampling_rate=0.5, #0.2
            syn_metric='mattes', 
            syn_sampling=10, #32
            reg_iterations=(20000, 20000, 20000),   #(40, 20, 0)
            #aff_iterations=(2100, 1200, 1200, 10), 
            #aff_shrink_factors=(0, 0, 0, 0),  #(6, 4, 2, 1),
            #aff_smoothing_sigmas=(0, 0, 0, 0),  #(3, 2, 1, 0)
            write_composite_transform=False, 
            random_seed=None, 
            verbose=False, 
            multivariate_extras=None, 
            restrict_transformation=None, 
            smoothing_in_mm=False
            #grad_step=0.05,  # gradient step, you can adjust based on your images
            #metric_weight=1,
            #radius_or_number_of_bins=100, # e.g., use for MI
            #sampling_strategy='Regular',
            #sampling_percentage=0.80,
            #number_of_iterations=[300],
            #convergence_threshold=1e-6,
            #convergence_window_size=100,
            #shrink_factors=[[0, 0, 0]],
            #smoothing_sigmas=[[0, 0, 0]]
        )
        
        
        if settings.reg_method == "Elastic":
            print("Elastic reg")
            # Stage 2: Affine Registration (optional, can be included for additional refinement)
            elastic_registration = ants.registration(
                fixed=fixed_image,
                moving=registration['warpedmovout'],
                type_of_transform='SyNAggro',
                mask_all_stages=False
                #grad_step=0.8, #gradient step size (not for all tx)
                #flow_sigma=3, #smoothing for update field
                #total_sigma=0, #smoothing for total field
                #aff_metric='mattes', #the metric for the affine part (GC, mattes, meansquares)
                #aff_sampling=64, #the nbins or radius parameter for the syn metric
                #aff_random_sampling_rate=0.8, #the fraction of points used to estimate the metric. this can impact speed but also reproducibility and/or accuracy.
                #syn_metric='mattes', #the metric for the syn part (CC, mattes, meansquares, demons)
                #syn_sampling=32, #the nbins or radius parameter for the syn metric
                #reg_iterations=(2000, 2000, 50), # vector of iterations for syn. we will set the smoothing and multi-resolution parameters based on the length of this vector.
                #aff_iterations=(100, 100, 50), #vector of iterations for low-dimensional (translation, rigid, affine) registration.
                #aff_shrink_factors=(0.5, 0, 0), #vector of multi-resolution shrink factors for low-dimensional (translation, rigid, affine) registration.
                #aff_smoothing_sigmas=(0.5, 0, 0), #vector of multi-resolution smoothing factors for low-dimensional (translation, rigid, affine) registration.
                #write_composite_transform=False, #Boolean specifying whether or not the composite transform (and its inverse, if it exists) should be written to an hdf5 composite file. This is false by default so that only the transform for each stage is written to file.
                #random_seed=None, #random seed to improve reproducibility. note that the number of ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS should be 1 if you want perfect reproducibility.
                #verbose=False, #request verbose output (useful for debugging)
                #multivariate_extras=None, #list of additional images and metrics which will trigger the use of multiple metrics in the registration process in the deformable stage. Each multivariate metric needs 5 entries: name of metric, fixed, moving, weight, samplingParam. the list of lists should be of the form ( ( “nameOfMetric2”, img, img, weight, metricParam ) ). Another example would be ( ( “MeanSquares”, f2, m2, 0.5, 0
                #, ( “CC”, f2, m2, 0.5, 2 ) ) . This is only compatible    with the SyNOnly or antsRegistrationSyN* transformations.
                #restrict_transformation=None, #This option allows the user to restrict the) – optimization of the displacement field, translation, rigid or affine transform on a per-component basis. For example, if one wants to limit the deformation or rotation of 3-D volume to the first two dimensions, this is possible by specifying a weight vector of ‘(1,1,0)’ for a 3D deformation field or ‘(1,1,0,1,1,0)’ for a rigid transformation. Restriction currently only works if there are no preceding transformations.
                #smoothing_in_mm=False #(boolean ; currently only impacts low dimensional registration)


                ) 
        
        
            #combine transforms
            fwd_transform = registration['fwdtransforms'] + \
                            elastic_registration['fwdtransforms']
            
        
        #warped_label = ants.apply_transforms(fixed=fixed_enhanced, moving=moving_label, transformlist=registration['fwdtransforms'], interpolator='genericLabel') #'nearestNeighbor')
        #warped_image = ants.apply_transforms(fixed=fixed_enhanced, moving=moving_image, transformlist=registration['fwdtransforms'], interpolator='linear')
        if settings.reg_method == "Rigid":
            warped_label = ants.apply_transforms(fixed=fixed_image, moving=moving_label, transformlist=registration['fwdtransforms'], interpolator='genericLabel') #'nearestNeighbor')
            warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration['fwdtransforms'], interpolator='linear')
            # The registered image becomes the new fixed image for the next iteration
            fixed_image = registration['warpedmovout'] 
        
        if settings.reg_method == "Elastic":
            warped_label = ants.apply_transforms(fixed=fixed_image, moving=moving_label, transformlist=fwd_transform, interpolator='genericLabel', defaultvalue=0) #'nearestNeighbor')
            warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=fwd_transform, interpolator='linear', defaultvalue=0)
            fixed_image = elastic_registration['warpedmovout']
                     
        
        registered_images.append(warped_image)
        registered_labels.append(warped_label)
       
        
    # Stack along a new first axis to create 4D array
    registered_images = np.stack(registered_images, axis=0)
    registered_labels = np.stack(registered_labels, axis=0)
    
    logger.info(f"Registration complete. \n")

    return registered_images, registered_labels


def tiff_to_nifti(input_path, output_path, logger):
    """
    Convert a TIFF image to NIfTI format using tifffile and nibabel.
    
    Parameters:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path where the output NIfTI file will be saved.
    """
    #Create output
    if not os.path.exists(output_path):
       os.makedirs(output_path)

    # Initialize reference to None
    reference_image = None
   
    # Loop over each image in the input directory
    for filename in os.listdir(input_path):
        if  filename.endswith(".tif"):  # 
            image_path = os.path.join(input_path, filename)
            # Read the TIFF image using tifffile
            data = imread(image_path)
                        
            #data = normalize_intensity(data)
            
            
            #data = adaptive_equalize(data)
            
            data = np.transpose(data, (2, 1, 0))
        
            # Create a new NIfTI image from the data
            # Note: This assumes no specific header information, so adjust as needed.
            new_image = nib.Nifti1Image(data, affine=np.eye(4))
            
            # Save the NIfTI image
            nib.save(new_image, os.path.join(output_path, os.path.splitext(filename)[0] + ".nii"))
             
def labels_to_nifti(input_path, output_path, logger):
    """
    Convert a TIFF image to NIfTI format using tifffile and nibabel.
    
    Parameters:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path where the output NIfTI file will be saved.
    """
    #Create output
    if not os.path.exists(output_path):
       os.makedirs(output_path)

    # Initialize reference to None
    reference_image = None
   
    # Loop over each image in the input directory
    for filename in os.listdir(input_path):
        if  filename.endswith(".tif"):  # 
            image_path = os.path.join(input_path, filename)
            # Read the TIFF image using tifffile
            data = imread(image_path)
            
            # If reference_image is None, it's the first image.
            if reference_image is None:
                #data = contrast_stretch(data, pmin=5, pmax=97)
                reference_image = data
                
            #else:
               # Match histogram of current image to the reference image
               #data = exposure.match_histograms(data, reference_image)
            
            #data = normalize_intensity(data)
            
            
            #data = adaptive_equalize(data)
            
            data = np.transpose(data, (2, 1, 0))
        
            # Create a new NIfTI image from the data
            # Note: This assumes no specific header information, so adjust as needed.
            new_image = nib.Nifti1Image(data, affine=np.eye(4))
            
            # Save the NIfTI image
            nib.save(new_image, os.path.join(output_path, os.path.splitext(filename)[0] + ".nii"))
             



def save_registered_images_to_tif(registered_images, output, logger):
    """
    Save a list of registered ANTs images to .tif format.
    
    Parameters:
        registered_images (list of ANTsImage): List of registered images from ANTsPy.
        output_folder (str): Path to the folder where the images will be saved.
    """

    # Ensure the output folder exists, if not, create it
    if not os.path.exists(output):
        os.makedirs(output)

    # Iterate through the registered images and save each to .tif format
    for idx, img in enumerate(registered_images, start=1):
        output_path = os.path.join(output, f"registered_image_{idx}.tif")
        ants.image_write(img, output_path)




def save_nifti_to_folder(input_path, output_path):
    """
    Save a NIfTI file to a specified directory and display its slices for verification.
    
    Parameters:
        input_path (str): Path to the input NIfTI file.
        folder_path (str): Directory where the NIfTI file will be saved.
    """
    
    # Ensure the directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define the destination path
    dest_path = os.path.join(output_path, os.path.basename(input_path))
    
    # Copy the NIfTI file to the destination directory
    os.replace(input_path, dest_path)
    
    # Load the image for visualization
    img = nib.load(dest_path)
    data = img.get_fdata()

    # Display slices
    # For simplicity, we're showing a slice from the middle of each axis
    axes = ['X', 'Y', 'Z']
    slice_indices = [data.shape[i] // 2 for i in range(3)]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ax in enumerate(axs):
        if i == 0:
            ax.imshow(data[slice_indices[i], :, :], cmap="gray")
        elif i == 1:
            ax.imshow(data[:, slice_indices[i], :], cmap="gray")
        else:
            ax.imshow(data[:, :, slice_indices[i]], cmap="gray")
        
        ax.set_title(f'Slice along {axes[i]} at index {slice_indices[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
    
"""
image = "D:/Project_Data/SpineAnalysis/Tracking_of_Spines/test_segment_man_clean/registered_labels/"

spines = spine_tracking(image)


"""

def spine_tracking(input_dir):
    data = imread(input_dir+"4d.tif")
    spines = data == 1
    
    labeled_spines = label(spines)
    
    #properties = regionprops(labeled_spines)
    #centroids = [prop.centroid for prop in properties]
    
    imwrite(input_dir+"labeled_spines.tif", labeled_spines.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': 1, 'unit': 'um','axes': 'TZYX', 'mode': 'composite'})
    
    





def link_objects(centroids_frame1, centroids_frame2):
    links = {}
    for i, centroid1 in enumerate(centroids_frame1):
        distances = [distance.euclidean(centroid1, centroid2) for centroid2 in centroids_frame2]
        closest = np.argmin(distances)
        links[i] = closest
    return links
    return data

"""
Overlap-Based Linking: Instead of just using centroids, consider linking objects 
based on overlap between frames.
Advanced Linking Algorithms: Consider more sophisticated 
algorithms for linking, such as the Hungarian method, especially 
when dealing with large numbers of objects and potential ambiguities in linking.

Handling Splitting & Merging: In many biological contexts, objects (like cells)
 can divide or merge. Handling these events requires more advanced tracking algorithms.
 
Additional Libraries: Libraries like trackpy offer more advanced features for 
particle tracking in 2D and can be adapted for 3D with some effort.
For real-world applications, especially in complex datasets, tracking 3D 
objects over time can be a significant challenge, and you might need specialized 
software or further advanced algorithms.
"""



def read_and_combine_images(input_dir, stack_axis =0):
    # Using pathlib to find all .tif files in the folder
    files = list(Path(input_dir).glob('*.tif'))

    # Read each tif file into a list of numpy arrays
    images = [imread(str(file)) for file in files]

    # Check if the images are of the same shape
    if len(set([image.shape for image in images])) != 1:
        raise ValueError("All images must have the same shape - xyz and channels.")

    # Stack the 3D images into a 4D numpy array
    stacked_images = np.stack(images, axis=stack_axis)
    
    return stacked_images

def normalize_intensity(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def contrast_stretch(image, pmin=2, pmax=98):
    p2, p98 = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(p2, p98))

def adaptive_equalize(image):
    return exposure.equalize_adapthist(image, clip_limit=0.03)

def match_histogram(source_image, reference_image):
    matched_image = exposure.match_histograms(source_image, reference_image)
    return matched_image
