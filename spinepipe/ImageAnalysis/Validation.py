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
        
        
        logger.info(f"  Ground truth data has shape {gt.shape}")
        logger.info(f"  Analysis output data has shape {output.shape}")
        
     
        #### GROUND TRUTH SPINES --- UPDATE BELOW LINES TO FUNCTIONS FOR READABILITY!!!
        spines = (gt == 1).astype(np.uint8)
        dendrites = (gt == 2).astype(np.uint8)
        #soma = (image == 3).astype(np.uint8)
        spines_orig = spines * 65535
        
        soma = (gt==3)
        if np.max(soma) == 5:
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
        
           
        logger.info(f"   Processing {filt_dendrites} dendrites...")
        
        dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites)) #invert neuron mask to get outside distance  
        dendrites = dendrites.astype(np.uint8)
        skeleton = morphology.skeletonize_3d(dendrites)
        
        soma_distance = ndimage.distance_transform_edt(np.invert(soma))
        
        
        
        #Detection
        logger.info("   Detecting spines...")
        spine_labels = imgan.spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
        
        max_label = np.max(spine_labels)
        
        #Measurements
        spine_table, gt_spines = imgan.spine_measurementsV2(dendrites, spine_labels, 1, max_label, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, analysis_files[file], logger)

        dendrite_length = np.sum(skeleton == 1)

        dendrite_volume = np.sum(dendrites ==1)
          
       
        gt_dendrite_length = dendrite_length
        gt_dendrites =  dendrites 
 
        
        #### OUTPUT SPINES --- UPDATE TO FUNCTIONS
        spines = (output == 1).astype(np.uint8)
        dendrites = (output == 2).astype(np.uint8)
        #soma = (image == 3).astype(np.uint8)
        spines_orig = spines * 65535
        
        soma = (gt==3)
        if np.max(soma) == 5:
            soma_distance = soma
        else:
            soma_distance = ndimage.distance_transform_edt(np.invert(soma))
        
        #dendrite_labels = measure.label(dendrites)
        #fitler out small dendites
        dendrite_labels, num_detected = ndimage.label(dendrites)
                
        # Calculate volumes and filter
        dend_vols = ndimage.sum_labels(dendrites, dendrite_labels, index=range(1, num_detected + 1))

        large_dendrites = dend_vols >= settings.min_dendrite_vol

        # Create new dendrite binary
        dendrites = np.isin(dendrite_labels, np.nonzero(large_dendrites)[0] + 1).astype(bool)

        filt_dendrites = np.max(measure.label(dendrites))
        
        logger.info(f"  {filt_dendrites} of {num_detected} detected dendrites larger than minimum volume threshold of {settings.min_dendrite_vol} voxels")
        
           
        logger.info(f"   Processing {filt_dendrites} dendrites...")
        
        dendrite_distance = ndimage.distance_transform_edt(np.invert(dendrites)) #invert neuron mask to get outside distance  
        dendrites = dendrites.astype(np.uint8)
        skeleton = morphology.skeletonize_3d(dendrites)
        
        soma_distance = ndimage.distance_transform_edt(np.invert(soma))
        
        
        
        #Detection
        logger.info("   Detecting spines...")
        spine_labels = imgan.spine_detection(spines, settings.erode_shape, settings.remove_touching_boarders, logger) #binary image, erosion value (0 for no erosion)
        
        max_label = np.max(spine_labels)
        
        #Measurements
        spine_table, output_spines = imgan.spine_measurementsV2(dendrites, spine_labels, 1, max_label, settings.neuron_channel, dendrite_distance, soma_distance, settings.neuron_spine_size, settings.neuron_spine_dist, settings, locations, analysis_files[file], logger)

        dendrite_length = np.sum(skeleton == 1)

        dendrite_volume = np.sum(dendrites ==1)
          
       
        output_dendrite_length = dendrite_length
        output_dendrites =  dendrites 
        
        
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
        
        #logger.info(f"{gt_spines.shape}")
        
        gt_spine_volume = np.sum(gt_spines >=1)
        #logger.info(f"{gt_spine_volume}")
        comp_spine_table.insert(12, 'gt_total_spine_vol', gt_spine_volume)
        output_spine_volume = np.sum(output_spines >=1)
        comp_spine_table.insert(13, 'output_total_spine_vol', output_spine_volume)
        comp_spine_table.insert(14, 'total_spine_iou', spine_iou)
        comp_spine_table.insert(15, 'total_spine_vol_difference', gt_spine_volume - output_spine_volume)
        
        
        comp_spine_table.insert(16, 'gt_dendrite_length', gt_dendrite_length)
        comp_spine_table.insert(17, 'output_dendrite_length', output_dendrite_length)
        comp_spine_table.insert(18, 'dendrite_length_difference', gt_dendrite_length - output_dendrite_length)
        
        
        gt_dendrite_volume = np.sum(gt_dendrites ==1)
        comp_spine_table.insert(19, 'gt_dendrite_vol', gt_dendrite_volume)
        output_dendrite_volume = np.sum(output_dendrites ==1)
        comp_spine_table.insert(20, 'output_dendrite_vol', output_dendrite_volume)
        comp_spine_table.insert(21, 'total_dendrite_iou', dendrite_iou)
        comp_spine_table.insert(22, 'dendrite_vol_difference', gt_dendrite_volume - output_dendrite_volume)
                                                          
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
    #TP_table = gt_table[(gt_table['mean_intensity'] > 0.75)]
    #TP_iou75 = TP_table.shape[0]
   
    spine_table.insert(7, 'TruePos_IoU50', TP_iou50)
    #spine_table.insert(8, 'TruePos_IoU75', TP_iou75)
    
    
    
    
    FP_table = output_table[(output_table['mean_intensity'] < 0.5)]
    FP_iou50 = FP_table.shape[0]
    spine_table.insert(8, 'FalsePos_IoU50', FP_iou50)
    
    #FalseNeg
    spine_table.insert(9, 'FalseNeg_IoU50', gt_spines_filtered - TP_iou50)
    #spine_table.insert(11, 'FalseNeg_IoU75', gt_spines_filtered - TP_iou75)
    
    #GT spine number
    
    #output spine number
    
    #gt spines detected
    
    #spine precision (TP / TP + FP)
    spine_precision_iou50 = TP_iou50 / (TP_iou50 + FP_iou50)
    #spine_precision_iou75 = TP_iou75 / (TP_iou75 + FP_iou50)
    spine_table.insert(10, 'spine_precision_IoU50', spine_precision_iou50)
    #spine_table.insert(13, 'spine_precision_IoU75', spine_precision_iou75)
    #spine_pre_iou050 - 50%
    #spine_pre_iou075 - strict        
    
    #spine recall (TP / TP + FN)
    spine_recall_iou50 = TP_iou50 / (TP_iou50 + (gt_spines_filtered - TP_iou50))
    #spine_recall_iou75 = TP_iou75 / (TP_iou75 + (gt_spines_filtered - TP_iou75))
    spine_table.insert(11, 'spine_recall_IoU50', spine_recall_iou50)
    #spine_table.insert(15, 'spine_recall_IoU75', spine_recall_iou75) 
  
    return spine_table


    
def IoU_calc(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    

