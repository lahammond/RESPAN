# SpinePipe
An efficient and accessible spine analysis pipeline leveraging the latest advancements in U-Net segmentation and GPU processing. 
To enable ease of use, this pipeline is made available as python code and as a standalone installation for Windows (requires NVIDIA GPU and CUDA)

Developed in collaboration with the Polleux Lab (Zuckerman Institute, Columbia University).

## SpinePipe
- GUI based automatic segmentation of spines and dendrites
- 3D morphological and intensity measurements
- 3D spatial relationships
- tabular and image outputs, including validation images, to ensure accuracy
- Spine Arrays - extract each spine as a 3D volume for visualization and further machine learning training of spine morphological features and spine classification

## SpinePipe Validation Tool
- perform automated quantitative assessment of the performance of the trained U-Net model and SpinePipe pipeline
- validate and readily evaluate improvements to the pipeline models using ground-truth data

## SpinePipe Model Training Tool (soon)
- simplified GUI-based tool to facilitate training of new nnUnet models

## Future updates:
- spine neck analysis
- track spine morphology and signal intensity over time
