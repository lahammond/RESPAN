# -*- coding: utf-8 -*-
"""
Image Analysis tools and functions for spine analysis
==========


"""
__title__     = 'spinpipe'
__version__   = '0.9.2'
__date__      = "25 Sept, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'


# Code to register timelapse images and then track spines

import os
import numpy as np 

import ants

import nibabel as nib
from tifffile import imread, imwrite

import matplotlib.pyplot as plt

from pathlib import Path

from skimage import img_as_uint
from skimage import exposure
from skimage.measure import label
from skimage.measure import regionprops
from scipy.spatial import distance

#inputdir = "D:/Project_Data/SpineAnalysis/Tracking_of_Spines/Segmentation_Mask/"

inputdir = "D:/Project_Data/SpineAnalysis/Tracking_of_Spines/test_segment_man_clean/"


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

def save_as_4d_tiff(images, outputdir, filename):
    """
    Save a 4D numpy array as a TIFF image.
    
    Parameters:
        array (np.array): 4D numpy array of image data.
        outputpath (str): Path where the 4D TIFF image will be saved.
    """
    #Create output
    if not os.path.exists(outputdir):
       os.makedirs(outputdir)
       
    # Convert each ANTs image to a numpy array and stack to form a 4D array
    stacked_array = np.stack([img.numpy() for img in images], axis=0)
    print(stacked_array.shape)
    reordered_array = np.moveaxis(stacked_array, [0, 1, 2, 3], [0, 3, 2, 1])
    #add an axis for C at position 0
    final_array = reordered_array[:, :, np.newaxis,  :, :]
    
    print(final_array.shape)
    
    #final_array = img_as_uint(final_array)
    
    imwrite(outputdir+filename, final_array.astype(np.uint16), imagej=True, photometric='minisblack',
            metadata={'spacing': 1, 'unit': 'um','axes': 'TZCYX', 'mode': 'composite'})
    
    return(final_array)


def serial_registration_from_folder_rigid(input_path):
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

    for i in range(1, len(images_list)):
        print(f"Processing: {images_list[i]}")
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

    return registered_images


def serial_registration_of_images_and_labels(img_input_path, labels_input_path):
    """
    Serially register a list of images using rigid body registration with ANTsPy.
    
    Parameters:
        images_list (list of str): List of image paths in order they should be registered.
    
    Returns:
        list of ANTsImage: List of registered images.
    """
    
    # Sort filenames to ensure the correct order
    images_list = sorted([os.path.join(img_input_path, filename) for filename in os.listdir(img_input_path) if filename.endswith('.nii')])

    labels_list = sorted([os.path.join(labels_input_path, filename) for filename in os.listdir(labels_input_path) if filename.endswith('.nii')])

    if not images_list:
        raise ValueError("No images found in the provided folder.")

    
    registered_images = []
    registered_labels = []
    # The first image is considered the fixed image for the first registration
    fixed_image = ants.image_read(images_list[0])
    fixed_label = ants.image_read(labels_list[0])
    
    registered_images.append(fixed_image)
    registered_labels.append(fixed_label)

    for i in range(1, len(images_list)):
        print(f"Processing: {images_list[i]}")
        moving_image = ants.image_read(images_list[i])
        moving_label = ants.image_read(labels_list[i])
        
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
        
          
        
        warped_label = ants.apply_transforms(fixed=fixed_image, moving=moving_label, transformlist=registration['fwdtransforms'], interpolator='nearestNeighbor')
        
        # The registered image becomes the new fixed image for the next iteration
        fixed_image = registration['warpedmovout']
        
        registered_images.append(fixed_image)
        registered_labels.append(warped_label)
        
        
        
    # Stack along a new first axis to create 4D array
    registered_images = np.stack(registered_images, axis=0)
    registered_labels = np.stack(registered_labels, axis=0)

    return registered_images, registered_labels


def tiff_to_nifti(input_path, output_path):
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
                data = contrast_stretch(data, pmin=5, pmax=97)
                reference_image = data
                
            else:
               # Match histogram of current image to the reference image
               data = exposure.match_histograms(data, reference_image)
            
            #data = normalize_intensity(data)
            
            
            #data = adaptive_equalize(data)
            
            data = np.transpose(data, (2, 1, 0))
        
            # Create a new NIfTI image from the data
            # Note: This assumes no specific header information, so adjust as needed.
            new_image = nib.Nifti1Image(data, affine=np.eye(4))
            
            # Save the NIfTI image
            nib.save(new_image, os.path.join(output_path, os.path.splitext(filename)[0] + ".nii"))
             
def labels_to_nifti(input_path, output_path):
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
             



def save_registered_images_to_tif(registered_images, output):
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
