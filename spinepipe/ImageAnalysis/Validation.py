# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'spinpipe'
__version__   = '0.9.0'
__date__      = "25 July, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'


import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import measure, morphology
from spinepipe.ImageAnalysis.ImageAnalysis import spine_detection


def validate_analysis(labels1, labels2, settings, locations, logger):
    logger.info("Validating spine and dendrite detection...")
    
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
    
    for file in range(len(analysis_files)):
        logger.info(f' Comparing image pair {file+1} of {len(analysis_files)} \n  Ground Truth Image:{gt_files[file]} \n  Analysis Output Image:{analysis_files[file]}')
        
        gt = imread(labels1 + gt_files[file])
        output = imread(labels2 + analysis_files[file])
        
        logger.info(f"  Ground truth data has shape {gt.shape}")
        logger.info(f"  Analysis output data has shape {output.shape}")
        
        #image = check_image_shape(image, logger) 
        
        #neuron = image[:,settings.neuron_channel-1,:,:]

        
        gt_spines = (gt == 1).astype(np.uint8)
        gt_dendrites = (gt == 2).astype(np.uint8)
        #soma = (image == 3).astype(np.uint8)
        spines = (output == 1).astype(np.uint8)
        dendrites = (output == 2).astype(np.uint8)
        
        gt_skeleton = morphology.skeletonize_3d(gt_dendrites)
        skeleton = morphology.skeletonize_3d(dendrites)
        
        #Detection
        logger.info(" Detecting spines...")
        gt_spine_labels = spine_detection(gt_spines, 10 ** 3, logger) #value used to remove small holes
        spine_labels = spine_detection(spines, 10 ** 3, logger) #value used to remove small holes
    
        #shorten filename
        filename = analysis_files[file].replace('.tif', '')    
    
        #create empty df
        spine_table = pd.DataFrame({'Filename': [filename]})
        
        
        # update summary with additional metrics
        spine_table.insert(1, 'res_XY', settings.input_resXY)  
        spine_table.insert(2, 'res_Z', settings.input_resZ)
        
        #Measurements
        spine_table = spine_comparison(gt_spine_labels, spine_labels, settings.neuron_spine_size, spine_table, logger)
                                                          #soma_mask, soma_distance, )
        
        
        
        
        #IoU
        spine_iou = IoU_calc(gt_spines, spines)
        
        dendrite_iou = IoU_calc(gt_dendrites, dendrites)
        
        
        
        gt_spine_volume = np.sum(gt_spines ==1)
        spine_table.insert(16, 'gt_total_spine_vol', gt_spine_volume)
        spine_volume = np.sum(spines ==1)
        spine_table.insert(17, 'output_total_spine_vol', spine_volume)
        spine_table.insert(18, 'total_spine_iou', spine_iou)
        spine_table.insert(19, 'total_spine_vol_difference', gt_spine_volume - spine_volume)
        
        
        gt_dendrite_length = np.sum(gt_skeleton == 1)
        spine_table.insert(20, 'gt_dendrite_length', gt_dendrite_length)
        dendrite_length = np.sum(skeleton == 1)
        spine_table.insert(21, 'output_dendrite_length', dendrite_length)
        spine_table.insert(22, 'dendrite_length_difference', gt_dendrite_length - dendrite_length)
        
        
        gt_dendrite_volume = np.sum(gt_dendrites ==1)
        spine_table.insert(23, 'gt_dendrite_vol', gt_dendrite_volume)
        dendrite_volume = np.sum(dendrites ==1)
        spine_table.insert(24, 'output_dendrite_vol', dendrite_volume)
        spine_table.insert(25, 'total_dendrite_iou', dendrite_iou)
        spine_table.insert(26, 'dendrite_vol_difference', gt_dendrite_volume - dendrite_volume)
                                                          
        #append to summary
        # Append to the overall summary DataFrame
        spine_summary = pd.concat([spine_summary, spine_table], ignore_index=True)
        
        
    
        logger.info(f"Spine comparison complete for file {analysis_files[file]}\n")
    spine_summary.to_csv(locations + 'Analysis_Evaluation.csv',index=False) 
    logger.info("\nSpine comparison complete.\n")

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
    

