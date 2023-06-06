# spine-analysis
Python tool for spine detection and analysis

- We can come up with a better name for the pipeline once it takes shape!

Currently:
- initial notebook discussed in meeting
- Resnet34 model - https://www.dropbox.com/s/8snm87e0pgu1thv/2023_05_spine_2D%20256x256_resnet34_V3_2.h5?dl=0

To include/update:
- simplified notebook for training sm u-net
- simplified notebook for segmenting images with u-net and performing basic analysis of spines
- Test performance of nnUnet - switch to this method for training and inference if better (initial tests look good!) - https://www.nature.com/articles/s41592-020-01008-z
- segmentation of dendrite (3D Unet, binary mask and skeleton outputs)
- automatic isolation of primary dendrite and exclude other dendrites/spines (size/length followed by distance transform)
- detection of spine neck
- Extract spine MIPs or 3D volumes for classification into spine type (mushroom, stubby, thin, filopodia...) 
- GUI for selecting folders, models, and analysis parameters 
