# RESPAN: An Automated Pipeline for Accurate Dendritic Spine Mapping with Integrated Image Restoration.
<p align="center">
  <img src="images/Panel 1.jpg" alt="RESPAN" width="600">
</p>
Restoration Enhanced SPine And Neuron (RESPAN) Analysis enables robust, accurate, and unbiased quantification of neurons and dendritic spines in a diverse range of imaging modalities. While developing this pipeline, emphasis was placed on ensuring an efficient and accessible pipeline that leverages the latest advancements in image restoration, image segmentation, and GPU processing. 
</p>
For ease of use, this pipeline is made available as a standalone application for Windows.</p> Please note this software requires a computer with an NVIDIA GPU (min 12GB TitanX recommended) and sufficient RAM (min 128GB recommended). Code is also available to run from Python.</p></p>
Developed in collaboration with the Polleux Lab (Zuckerman Institute, Columbia University).
</p>
<p align="center">
  <img src="images/Panel 2.jpg" alt="RESPAN" width="600">
</p>

## Download RESPAN
Download the latest version of the RESPAN Windows executable [here](). This file is zipped using [7zip](https://www.7-zip.org/). </p> Please allow a minimum of 20GB of disk space for the software and ensure there is sufficient space for processing your data. RESPAN includes lossless compression of image files to ensure a minimal footprint for generated results and validation images.

## Advantages of RESPAN
*	Comprehensive Integration: RESPAN uniquely integrates image restoration, axial resolution enhancement, and deep learning-based segmentation into a single, user-friendly application.
*	3-Dimensional Analysis: 3D information is efficiently utilized at all stages of the pipeline, ensuring improved performance over approaches limited to 2D or a combination of 2D and 3D techniques for quantification.
*	In Vivo Spine Tracking: RESPAN has been demonstrated to successfully track spines autonomously across time in 3D in challenging in vivo two-photon imaging conditions.
*	Increased Accuracy: By enhancing image quality prior to segmentation, RESPAN significantly improves the accuracy of spine detection and morphological measurements.
*	User-Friendly Deployment and Interface: A ready-to-run application with a graphical user interface allows users without programming skills to perform advanced analyses.
*	Built-in Validation Tools: RESPAN includes tools for validating results against ground-truth data, promoting scientific rigor and reproducibility.
*	Model Training: RESPAN includes tabs in the graphical user interface that allow training of CSBDeep, Self-Net, and nnU-Net models, which normally require separate environments using Tensorflow and PyTorch, removing a significant barrier to training and utilizing custom models. 

## RESPAN Publications
If you use RESPAN as part of your research, please cite our work using the reference below:

RESPAN has been used in the following publications:
has been used in the publications listed below:


## RESPAN Installation
Updating soon with information on how to create the environments for running RESPAN directly in Python.
