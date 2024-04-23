# -*- coding: utf-8 -*-
"""
Luke Hammond

Care processing batch script

19 December 2023

"""

import os
import numpy as np
from tifffile import imread, imwrite
from csbdeep.models import CARE
from skimage.util import view_as_blocks

model_path = r'D:\Dropbox\Github\spine-analysis\spinepipe\Models\Restoration\100xSil_xy65nm_z150nm_BFP_dendrite_decon_v1'

input_folder = r'D:\Project_Data\2024_03_SpinePipe Restore\input_unseen'
output_folder = os.path.join(input_folder, 'Restored')


patch_size = (64, 512, 512) # Faster to use fewer Z tiles and more XY tiles 2 x 4 x4 faster than 3 x 4 x 


#metadata for imwrite
metadata = {
    'spacing': 0.15,  # Z spacing
    'unit': 'um',
    'axes': 'ZYX'
}




model = CARE(config=None, name=model_path)

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process files that include all specified substrings and end with '.tif'
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'): #any(substr in filename for substr in filename_inclusions) and filename.endswith('.tif'):
        print(f"Processing file {filename}")
       
        file_path = os.path.join(input_folder, filename)
        image = imread(file_path)

        print(f"Image shape {image.shape}")
        print(" Padding...")
        
        # Calculate padding needed to make the image divisible by the patch size
        pad_widths = [(0, desired - current % desired) if current % desired != 0 else (0, 0) 
                      for current, desired in zip(image.shape, patch_size)]
        padded_image = np.pad(image, pad_widths, mode='symmetric')
        
        print(" Patching...")
        # Extract patches
        patches = view_as_blocks(padded_image, patch_size)
        restored_patches = np.zeros_like(patches)
        
        # Process each patch
        print(" Restoring...")
        total_patches = patches.shape[0] * patches.shape[1] * patches.shape[2]
        patch_count = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                for k in range(patches.shape[2]):
                    patch = patches[i, j, k]
                    restored_patch = model.predict(patch, axes='ZYX')
                    restored_patches[i, j, k] = restored_patch
                    patch_count += 1
                    print(f"Processed patch {patch_count} of {total_patches}", end='\r')
        
        print(" Reassembling and unpadding") 
        # Reassemble the processed patches
        restored_image = np.block([[[restored_patches[i, j, k] 
                                     for k in range(restored_patches.shape[2])]
                                    for j in range(restored_patches.shape[1])]
                                   for i in range(restored_patches.shape[0])])

        # Calculate the original image size before padding
        original_size = [length - pad_width[0] - pad_width[1] for length, pad_width in zip(padded_image.shape, pad_widths)]
        # Unpad the image to get back to the original size
        unpadded_restored_image = restored_image[:original_size[0], :original_size[1], :original_size[2]]

          
        output_path = os.path.join(output_folder, filename)
        
        imwrite(output_path, unpadded_restored_image, metadata=metadata, imagej=True)
        print(" Complete.")
        

