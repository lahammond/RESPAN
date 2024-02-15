# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'SpinePipe'
__version__   = '0.9.6'
__date__      = "23 December, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'


import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import measure, morphology
import spinepipe.ImageAnalysis.ImageAnalysis as imgan
from scipy import ndimage



def validate_analysis(labels1, labels2, settings, locations, logger):
    logger.info("Validating spine and dendrite detection...")
    logger.info(" ")
    
    labels1 = labels1+"/"
    labels2 =  labels2+"/"
    #labels 1 = ground truth dir
    #labels 2 = analysis output dirs
    
    #spines = 1
    #dendrites = 2
    #soma = 3
    
    gt_files = [file_i
             for file_i in os.listdir(labels1)
             if file_i.endswith('.tif')]
    gt_files = sorted(gt_files)
    
    analysis_files = [file_i
             for file_i in os.listdir(labels2)
             if file_i.endswith('.tif')]
    analysis_files = sorted(analysis_files)
    
    if len(gt_files) != len(analysis_files):
        raise RuntimeError("Lists are not of equal length.")
    
    spine_summary = pd.DataFrame()
    comp_spine_table = pd.DataFrame()
    comp_spine_summary = pd.DataFrame()

    
    for file in range(len(analysis_files)):
        
        logger.info(f' Comparing image pair {file+1} of {len(analysis_files)} \n  Ground Truth Image:{gt_files[file]} \n  Analysis Output Image:{analysis_files[file]}')
        
        gt = imread(labels1 + gt_files[file])
        output = imread(labels2 + analysis_files[file])
        
        min_dendrite_vol = round(settings.min_dendrite_vol / settings.input_resXY/settings.input_resXY/settings.input_resZ, 0)
        
        
        logger.info(f"  Ground truth data has shape {gt.shape}")
        logger.info(f"  Analysis output data has shape {output.shape}")
        
        #image = check_image_shape(image, logger) 
        
        #neuron = image[:,settings.neuron_channel-1,:,:]

        #### GROUND TRUTH SPINES --- UPDATE BELOW LINES TO FUNCTIONS FOR READABILITY!!!
        spines = (gt == 1).astype(np.uint8)
        all_dendrites = (gt == 2).astype(np.uint8)
        #soma = (image == 3).astype(np.uint8)
        spines_orig = spines * 65535
        
        dendrite_labels = measure.label(all_dendrites)
        
        all_filtered_spines = np.zeros_like(spines)
        all_skeletons = np.zeros_like(spines)
        
        all_filtered_spines_table = pd.DataFrame()
        
        all_dendrites = np.zeros_like(spines)
        total_dendrite_length = 0
        
        for dendrite in range(1, np.max(dendrite_labels)+1):
            logger.info(f" Detecting spines on dendrite {dendrite}...")
            dendrite_mask = dendrite_labels == dendrite
            
            if np.sum(dendrite_mask) > min_dendrite_vol:
                
            
                #Create Distance Map
    
                dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrite_mask)) #invert neuron mask to get outside distance  
                dendrite_mask = dendrite_mask.astype(np.uint8)
                all_dendrites = all_dendrites + dendrite_mask
                
                skeleton = morphology.skeletonize_3d(dendrite_mask)
                all_skeletons = skeleton + all_skeletons
               
                #Detection
                
                spine_labels = imgan.spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
                
                max_label = np.max(all_filtered_spines)
              
                #Measurements
                spine_table, summary, spines_filtered = imgan.spine_measurements(all_dendrites, spine_labels, dendrite, max_label, settings.neuron_channel, dendrite_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, gt_files[file], logger)
                                                                  #soma_mask, soma_distance, )
               
                #offset detected objects in image and add to all spines
                
               
                all_filtered_spines = all_filtered_spines + spines_filtered
                
    
                #remove detected spines from original spine image to prevent double counting
                spines[spines_filtered > 0] = 0
                            
    
                # update summary with additional metrics
                summary.insert(1, 'res_XY', settings.input_resXY)  
                summary.insert(2, 'res_Z', settings.input_resZ)
                dendrite_length = np.sum(skeleton == 1)
                total_dendrite_length += dendrite_length
                #summary.insert(3, 'Dendrite_ID', dendrite)
                summary.insert(3, 'dendrite_length', dendrite_length)
                dendrite_length_um = dendrite_length*settings.input_resXY
                summary.insert(4, 'dendrite_length_um', dendrite_length_um)
                dendrite_volume = np.sum(dendrite_mask ==1)
                summary.insert(5, 'dendrite_vol', dendrite_volume)
                dendrite_volume_um3 = dendrite_volume*settings.input_resXY*settings.input_resXY*settings.input_resZ
                summary.insert(6, 'dendrite_vol_um3', dendrite_volume_um3)
                summary.insert(9, 'spines_per_um_length', summary['total_spines'][0]/dendrite_length_um)
                summary.insert(10, 'spines_per_um3_vol', summary['total_spines'][0]/dendrite_volume_um3)
                                                              
                #append to summary
                # Append to the overall summary DataFrame
                spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
                all_filtered_spines_table = pd.concat([all_filtered_spines_table, spine_table], ignore_index=True)
                
                
            else:
                logger.info(f"  Dendrite excluded. Volume smaller than threshold of {min_dendrite_vol} voxels")
        
       
        #relable the label image
        # Get unique labels while preserving the order
        unique_labels = np.unique(all_filtered_spines[all_filtered_spines > 0])
       
        # Create a mapping from old labels to new sequential labels
        label_mapping = {labelx: i for i, labelx in enumerate(unique_labels, start=1)}

        # Apply this mapping to the image to create a relabelled image
        gt_spines = np.copy(all_filtered_spines)
        for old_label, new_label in label_mapping.items():
            gt_spines[gt_spines == old_label] = new_label
        
        all_filtered_spines_table['label'] = all_filtered_spines_table['label'].map(label_mapping)
       # all_filtered_spines_table.to_csv(locations.tables + 'Detected_spines_'+files[file]+'.csv',index=False) 
        
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines_orig, relabelled_spines, dendrite_labels, all_skeletons, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        
        gt_total_dendrite_length = total_dendrite_length
        gt_dendrites =  all_dendrites 
        #Extract MIPs for each spine
        #spine_MIPs, spine_slices, filtered_spine_MIPs, filtered_spine_slices= create_spine_arrays_in_blocks(image, relabelled_spines, all_filtered_spines_table, settings.spine_roi_volume_size, settings, locations, files[file],  logger, settings.GPU_block_size)
        

        
        #### OUTPUT SPINES --- UPDATE TO FUNCTIONS
        spines = (output == 1).astype(np.uint8)
        all_dendrites = (output == 2).astype(np.uint8)
        #soma = (image == 3).astype(np.uint8)
        spines_orig = spines * 65535
        
        dendrite_labels = measure.label(all_dendrites)
        
        all_filtered_spines = np.zeros_like(spines)
        all_skeletons = np.zeros_like(spines)
        
        
        all_filtered_spines_table = pd.DataFrame()
        
        all_dendrites = np.zeros_like(spines)
        total_dendrite_length = 0
        
        for dendrite in range(1, np.max(dendrite_labels)+1):
            logger.info(f" Detecting spines on dendrite {dendrite}...")
            dendrite_mask = dendrite_labels == dendrite
            
            if np.sum(dendrite_mask) > min_dendrite_vol:
                
            
                #Create Distance Map
    
                dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrite_mask)) #invert neuron mask to get outside distance  
                dendrite_mask = dendrite_mask.astype(np.uint8)
                all_dendrites = all_dendrites + dendrite_mask
                
                skeleton = morphology.skeletonize_3d(dendrite_mask)
                all_skeletons = skeleton + all_skeletons
               
                #Detection
                
                spine_labels = imgan.spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
                
                max_label = np.max(all_filtered_spines)
              
                #Measurements
                spine_table, summary, spines_filtered = imgan.spine_measurements(all_dendrites, spine_labels, dendrite, max_label, settings.neuron_channel, dendrite_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, analysis_files[file], logger)
                                                                  #soma_mask, soma_distance, )
                
               
                #offset detected objects in image and add to all spines
                
               
                all_filtered_spines = all_filtered_spines + spines_filtered
                
    
                #remove detected spines from original spine image to prevent double counting
                spines[spines_filtered > 0] = 0
                            
    
                # update summary with additional metrics
                summary.insert(1, 'res_XY', settings.input_resXY)  
                summary.insert(2, 'res_Z', settings.input_resZ)
                dendrite_length = np.sum(skeleton == 1)
                total_dendrite_length += dendrite_length
                #summary.insert(3, 'Dendrite_ID', dendrite)
                summary.insert(3, 'dendrite_length', dendrite_length)
                dendrite_length_um = dendrite_length*settings.input_resXY
                summary.insert(4, 'dendrite_length_um', dendrite_length_um)
                dendrite_volume = np.sum(dendrite_mask ==1)
                summary.insert(5, 'dendrite_vol', dendrite_volume)
                dendrite_volume_um3 = dendrite_volume*settings.input_resXY*settings.input_resXY*settings.input_resZ
                summary.insert(6, 'dendrite_vol_um3', dendrite_volume_um3)
                summary.insert(9, 'spines_per_um_length', summary['total_spines'][0]/dendrite_length_um)
                summary.insert(10, 'spines_per_um3_vol', summary['total_spines'][0]/dendrite_volume_um3)
                                                              
                #append to summary
                # Append to the overall summary DataFrame
                spine_summary = pd.concat([spine_summary, summary], ignore_index=True)
                all_filtered_spines_table = pd.concat([all_filtered_spines_table, spine_table], ignore_index=True)
                
                
            else:
                logger.info(f"  Dendrite excluded. Volume smaller than threshold of {min_dendrite_vol} voxels")
        
       
        #relable the label image
        # Get unique labels while preserving the order
        unique_labels = np.unique(all_filtered_spines[all_filtered_spines > 0])
       
        # Create a mapping from old labels to new sequential labels
        label_mapping = {labelx: i for i, labelx in enumerate(unique_labels, start=1)}

        # Apply this mapping to the image to create a relabelled image
        output_spines = np.copy(all_filtered_spines)
        for old_label, new_label in label_mapping.items():
            output_spines[output_spines == old_label] = new_label
        
        all_filtered_spines_table['label'] = all_filtered_spines_table['label'].map(label_mapping)
       # all_filtered_spines_table.to_csv(locations.tables + 'Detected_spines_'+files[file]+'.csv',index=False) 
        
        #neuron_MIP = create_mip_and_save_multichannel_tiff([neuron, spines_orig, relabelled_spines, dendrite_labels, all_skeletons, dendrite_distance], locations.MIPs+"MIP_"+files[file], 'float', settings)
        output_total_dendrite_length = total_dendrite_length
        output_dendrites =  all_dendrites 
        
        
        
        ## Original Code from simple validation
    
    
        #shorten filename
        filename = analysis_files[file].replace('.tif', '')    
    
        #create empty df
        comp_spine_table = pd.DataFrame({'Filename': [filename]})
        
        
        # update summary with additional metrics
        comp_spine_table.insert(1, 'res_XY', settings.input_resXY)  
        comp_spine_table.insert(2, 'res_Z', settings.input_resZ)
        
        #Measurements
        comp_spine_table = spine_comparison(gt_spines, output_spines, settings.neuron_spine_size, comp_spine_table, logger)
                                                          #soma_mask, soma_distance, )
        
               
        #IoU
        spine_iou = IoU_calc(gt_spines, output_spines)
        
        dendrite_iou = IoU_calc(gt_dendrites, output_dendrites)
        
        
        
        gt_spine_volume = np.sum(gt_spines >=1)
        comp_spine_table.insert(16, 'gt_total_spine_vol', gt_spine_volume)
        output_spine_volume = np.sum(output_spines >=1)
        comp_spine_table.insert(17, 'output_total_spine_vol', output_spine_volume)
        comp_spine_table.insert(18, 'total_spine_iou', spine_iou)
        comp_spine_table.insert(19, 'total_spine_vol_difference', gt_spine_volume - output_spine_volume)
        
        
        comp_spine_table.insert(20, 'gt_dendrite_length', gt_total_dendrite_length)
        comp_spine_table.insert(21, 'output_dendrite_length', output_total_dendrite_length)
        comp_spine_table.insert(22, 'dendrite_length_difference', gt_total_dendrite_length - output_total_dendrite_length)
        
        
        gt_dendrite_volume = np.sum(gt_dendrites ==1)
        comp_spine_table.insert(23, 'gt_dendrite_vol', gt_dendrite_volume)
        output_dendrite_volume = np.sum(output_dendrites ==1)
        comp_spine_table.insert(24, 'output_dendrite_vol', output_dendrite_volume)
        comp_spine_table.insert(25, 'total_dendrite_iou', dendrite_iou)
        comp_spine_table.insert(26, 'dendrite_vol_difference', gt_dendrite_volume - output_dendrite_volume)
                                                          
        #append to summary
        # Append to the overall summary DataFrame
        comp_spine_summary = pd.concat([comp_spine_summary, comp_spine_table], ignore_index=True)
        
        
    
        logger.info(f"Spine comparison complete for file {analysis_files[file]}\n")
        
    comp_spine_summary.to_csv(locations + 'Analysis_Evaluation.csv',index=False) 
    logger.info("\nSpinePipe validation complete.\n")

def spine_comparison(gt, output, sizes, spine_table, logger):
    """ compares GT and output spines
    """
    gt_binary = gt >= 1
    output_binary = output >= 1
    
    #filter out small objects
    volume_min = sizes[0] #3
    volume_max = sizes[1] #1500?
    
    
    #Measure channel 1:
    logger.info(" Measuring spines...")
    gt_table = pd.DataFrame(
        measure.regionprops_table(
            gt,
            intensity_image=output_binary,
            properties=['label', 'area', 'mean_intensity'], #area is volume for 3D images
            )
        )
    
    output_table = pd.DataFrame(
       measure.regionprops_table(
           output,
           intensity_image=gt_binary,
           properties=['label', 'area', 'mean_intensity'], #area is volume for 3D images
           )
       )
    
    
    gt_spines = gt_table.shape[0]
    spine_table.insert(3, 'gt_total_spines', gt_spines)
    
    filtered_table = gt_table[(gt_table['area'] > volume_min) & (gt_table['area'] < volume_max) ]
    
    gt_spines_filtered = gt_table.shape[0]
    spine_table.insert(4, 'gt_total_spines_filtered', gt_spines_filtered)
    
    output_spines = output_table.shape[0]
    spine_table.insert(5, 'output_total_spines', output_spines)
    
    filtered_table = output_table[(output_table['area'] > volume_min) & (output_table['area'] < volume_max) ]  
    output_spines_filtered = output_table.shape[0]
    spine_table.insert(6, 'output_total_spines_filtered', output_spines_filtered)
    
    
    TP_table = gt_table[(gt_table['mean_intensity'] > 0.5)]
    TP_iou50 = TP_table.shape[0]
    TP_table = gt_table[(gt_table['mean_intensity'] > 0.75)]
    TP_iou75 = TP_table.shape[0]
   
    spine_table.insert(7, 'TruePos_IoU50', TP_iou50)
    spine_table.insert(8, 'TruePos_IoU75', TP_iou75)
    
    
    
    
    FP_table = output_table[(output_table['mean_intensity'] < 0.5)]
    FP_iou50 = FP_table.shape[0]
    spine_table.insert(9, 'FalsePos_IoU50', FP_iou50)
    
    #FalseNeg
    spine_table.insert(10, 'FalseNeg_IoU50', gt_spines_filtered - TP_iou50)
    spine_table.insert(11, 'FalseNeg_IoU75', gt_spines_filtered - TP_iou75)
    
    #GT spine number
    
    #output spine number
    
    #gt spines detected
    
    #spine precision (TP / TP + FP)
    spine_precision_iou50 = TP_iou50 / (TP_iou50 + FP_iou50)
    spine_precision_iou75 = TP_iou75 / (TP_iou75 + FP_iou50)
    spine_table.insert(12, 'spine_precision_IoU50', spine_precision_iou50)
    spine_table.insert(13, 'spine_precision_IoU75', spine_precision_iou75)
    #spine_pre_iou050 - 50%
    #spine_pre_iou075 - strict        
    
    #spine recall (TP / TP + FN)
    spine_recall_iou50 = TP_iou50 / (TP_iou50 + (gt_spines_filtered - TP_iou50))
    spine_recall_iou75 = TP_iou75 / (TP_iou75 + (gt_spines_filtered - TP_iou75))
    spine_table.insert(14, 'spine_recall_IoU50', spine_recall_iou50)
    spine_table.insert(15, 'spine_recall_IoU75', spine_recall_iou75) 
  
    return spine_table


    
def IoU_calc(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    

