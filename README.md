# RESPAN: Restoration Enhanced SPine And Neuron Analysis.
<p align="center">
  <img src="images/Panel 1.jpg" alt="RESPAN" width="600">
</p>
RESPAN is an end‑to‑end, GPU‑accelerated pipeline that restores, segments, and quantifies dendrites and dendritic spines in fluorescent microscopy images in a robust, accurate, and unbiased manner. While developing this pipeline, emphasis was placed on ensuring an efficient and accessible pipeline that leverages the latest advancements in content‑aware restoration, image segmentation, and GPU processing. 
</p>
For ease of use, RESPAN is available as both (i) a ready‑to‑run Windows application and (ii) Python scripts. 
</p> Please note that this software requires a computer with an NVIDIA GPU. </p>
Developed in collaboration with the Polleux Lab (Zuckerman Institute, Columbia University).
</p>
<p align="center">
  <img src="images/Panel 2.jpg" alt="RESPAN" width="600">
</p>

---

## ✨ Key features

- **All‑in‑one workflow** – restoration → segmentation → quantification → validation in a single interface.
- **True 3D analysis** – every stage uses volumetric data.
- **In‑vivo spine analysis** – robust to low SNR in two‑photon datasets and challenging samples.
- **Model training from the GUI** – train/finetune nnU‑Net, CARE‑3D or SelfNet without code.
- **Comprehensive and Automatic Results** – automatic generation of validation MIPs, 3D volumes, and comprehensive spatial/morphological statistics.
- **Built‑in validation** – compare ground truth datasets to RESPAN outputs to validate quantification.
- **Step-by-step tutorials** - view our introduction and tutorials for analysis and model training [here] (https://youtu.be/Q6zu6y5P6Mc)
- **Stand‑alone or scriptable** – run the GUI on Windows or from a Python environment.
- **Lossless compression** – gzip lossless compression of data ensures a minimal footprint for generated results and validation images.

---

## 💻 System requirements

| | **Minimum Recommended** | **Recommended** |
|---|---|---|
| OS | Windows 10/11 ×64 | Windows 10/11 ×64 |
| GPU | NVIDIA ≥ 8 GB VRAM | NVIDIA RTX 4090 (24 GB) |
| RAM | 32 GB | 128–256 GB |
| Storage | HDD | SSD |

> *RESPAN should work for NVIDIA GPUs with less than 8GB, but this has not been tested.<br>
> *RESPAN implements data chunking and tiling, but for some steps, larger images currently necessitate increased RAM requirements. <br>
> *Please refer to the table at the end of this document for further performance testing information.

---

## 🚀 Quick start (Windows GUI)

If you need help getting started, please refer to our video tutorial with chapters linked below:<br>
    • [Introduction to RESPAN and Image Segmentation](https://www.youtube.com/watch?v=Q6zu6y5P6Mc)<br>
    • [Installing RESPAN](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=1513s)<br>
    • [Navigating the RESPAN GUI](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=1787s)<br>
    • [Example use of RESPAN](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=2318s)<br>
    • [Understanding RESPAN Outputs](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=2677s)<br>
    • [Training CARE Models in RESPAN](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=3122s)<br>
    • [Training SelfNet Models in RESPAN](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=3390s)<br>
    • [Using Restoration Models during RESPAN Analysis](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=3515s)<br>
    • [Training an nnU-Net Model using RESPAN](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=3768s)<br>
<br>    
1. **Download**  
   • Latest RESPAN release &nbsp;→&nbsp; [Windows Application](https://drive.google.com/drive/folders/1MUFsFDKPBON9v7A3ZRJSUd6qjPTuI9G1)
     if required, earlier versions of RESPAN can be found in our archive [here](https://drive.google.com/drive/folders/1RMCacdm_MheJ31bbU6NSI51Ho4pHPXch?usp=drive_link)
   • RESPAN Analysis Settings file &nbsp;→&nbsp; [here] https://drive.google.com/file/d/1sZoBfViD62nNu-9FYWtYMHtLq6Hwjwhk/view?usp=drive_link
   • Pre‑trained models &nbsp;→&nbsp; see table below<br>
3. **Install**  
   ▸ Unzip RESPAN.zip with [7zip](https://www.7-zip.org/)<br>
   ▸ Double‑click RESPAN.exe (first run may require 1-2 min to initialize)<br>
4. **Prepare your data**  
   ```text
   MyExperiment/
   ├── Animal_A/
   │   ├── dendrite0.tif
   │   ├── dendrite1.tif   
   │   └── Analysis_Settings.yml (example file provided [here](https://drive.google.com/file/d/1sZoBfViD62nNu-9FYWtYMHtLq6Hwjwhk/view?usp=drive_link) 
   └── Animal_B/
       ├── dendrite0.tif
       ├── ...   
       └── Analysis_Settings.yml
   ```
   *Copy **Analysis_Settings.yml** into every sub‑folder (stores resolution, advanced settings, and allows batch processing. Default settings suit most experiments, with editing only required when using advanced functionality and image restoration).
5. **Run**  
   • Select the *parent* folder (e.g. "MyExperiment") in the GUI  
   • Update analysis settings
   • Click **Run** – a 100 MB stack processes in ≈3 min on an RTX 4090
6. **Inspect outputs**  
   | Folder | Contents |
   |---|---|
   | `Tables/` | Per‑image CSVs (`Detected_spines_*.csv`) + experiment summary |
   | `Validation_Data/` | MIPs & volumes for QA (input, labels, skeleton, distance) |
   | `SWC_files/` | Neuron/dendrite traces from Vaa3D |
   | `Spine_Arrays/` | Cropped 2D maximum intensity projections and 3D stacks centered around every spine |

---

## 🖼️ Input data & considerations

* **File format** – RESPAN currently accepts 2D/3D **TIFF** files.
* **Conversion macro** – use the supplied [Fiji + OMERO‑BioFormats macro](https://github.com/lahammond/RESPAN/blob/main/RESPAN/ImageJ_Macros/Tif_Batch_2024.ijm) to batch‑convert ND2/CZI/LIF, etc.
* **Model specificity** – image‑restoration models (CARE & SelfNet) must match the modality & resolution being analyzed; mismatches can hallucinate or erase features. We strongly encourage retraining specific models for the microscope, objective, and resolution being used. RESPAN adapts input data to our pretrained segmentation models, and good results are likely without retraining, but we recommend using these first-pass results to fine-tune or train application-specific models 
* **Zarr support** – Internally, RESPAN has added OME-Zarr generation to support larger datasets, with future updates intending to utilize these files with Dask.
  
---

## 🛠️ Advanced usage: training new models

| Task | GUI Tab | Typical time | Tutorial link |
|------|---------|--------------|----------------|
| Segmentation (nnU‑Net) | **nnU‑Net Training** | 12–24 h | [tutorial](https://www.youtube.com/watch?v=Q6zu6y5P6Mc&t=3768s) |
| Image restoration (CARE‑3D) | **CARE Training** | 3–5 h | tutorial |
| Axial resolution (SelfNet) | **SelfNet Training** | ≤2 h | tutorial |

Detailed protocols – including data organisation and annotation tips – are in the [User Guide](https://github.com/lahammond/RESPAN/blob/main/RESPAN%20Guide%202025.pdf).

---

## 🎯 Pre‑trained segmentation models
  
| Segmentation Model | Download | Modality | Resolution | Annotations | Details |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Model 1A | [download](https://drive.google.com/drive/folders/1j9XQi4x1-IsRvln_-dzR-vF94cuV35RZ?usp=drive_link) | Spinning disk and Airyscan/laser scanning confocal microscopy | 65 x 65 x 150nm | spines, dendrites, and soma | 224 datasets, including restored and raw data and additional augmentation |
| Model 2B | [download](https://drive.google.com/drive/folders/1LTEO4aApPfTeTDEECMC-QJCpwWkL_C6w?usp=drive_link) | Spinning disk and Airyscan/laser scanning confocal microscopy | 65 x 65 x 150nm | spines core & shell, dendrites, axons, and soma | 44 datasets, including restored and raw data and additional augmentation |
| Model 2 | [download](https://drive.google.com/drive/folders/1MTSU7acFDZOUs9q1t_0gwU9DIbckYugi?usp=drive_link) | Spinning disk confocal microscopy  | 65 x 65 x 65nm | spines, necks, dendrites, and soma | isotropic model, 7 datasets, no augmentation |
| Model 3 |  [download](https://drive.google.com/drive/folders/1-KQc-Tzpk1dn3fgAln-KJrIIwM6amVFG?usp=drive_link) |Two-photon in vivo confocal microscopy  | 102 x 102 x 1000nm | spines and dendrites | 908 datasets, additional augmentation |

For detailed protocols using RESPAN, please refer to [our manuscript.](https://www.biorxiv.org/content/10.1101/2024.06.06.597812v2)

---

## 🔍 Validation workflow
This procedure guides you through validating RESPAN's segmentation outputs against a ground truth dataset. If you have not generated a ground truth annotation dataset, please refer to the notes below on creating annotations as a guide on how to generate these annotations for your specific datasets before you proceed.
CRITICAL: Ground truth annotations and the corresponding raw data volumes intended for validation testing should not be used in the training of nnU-Net models they are intended to test.

1. Open the **Analysis Validation** tab.  
2. Select the "analysis output directory" -  this is the `Validation_Data\Segmentation_labels` folder created by RESPAN during analysis
3. Select the "ground truth data directory" - this is a folder containing ground truth annotations for the data analyzed by RESPAN  
4. Adjust detection thresholds if needed
5. Click **Run**.  
6. Metrics are saved to `Analysis_Evaluation.csv`.
   
---

## 📚 Publications

If RESPAN assisted your research, please cite our work using the reference below:
If you use RESPAN as part of your research, please cite our work using the reference below:</p>
Sergio B. Garcia, Alexa P. Schlotter, Daniela Pereira, Franck Polleux, Luke A. Hammond. (2024) RESPAN: An Automated Pipeline for Accurate Dendritic Spine Mapping with Integrated Image Restoration. bioRxiv. doi: https://doi.org/10.1101/2024.06.06.597812</p></p>

RESPAN is already supporting peer-reviewed studies:
* Baptiste Libé-Philippot, Ryohei Iwata, Aleksandra J. Recupero, Keimpe Wierda, Sergio Bernal Garcia, Luke Hammond, Anja van Benthem, Ridha Limame, Martyna Ditkowska, Sofie Beckers, Vaiva Gaspariunaite, Eugénie Peze-Heidsieck, Daan Remans, Cécile Charrier, Tom Theys, Franck Polleux, Pierre Vanderhaeghen (2024)
Synaptic neoteny of human cortical neurons requires species-specific balancing of SRGAP2-SYNGAP1 cross-inhibition. Neuron. https://doi.org/10.1016/j.neuron.2024.08.021.

---
## 🛠️ Advanced usage: creating environments for use in Python
Main development environment:
1. mamba create -n respandev python=3.9 scikit-image pandas "numpy=1.23.4" nibabel pyinstaller ipython pyyaml pynvml numba dask dask-image ome-zarr zarr memory_profiler trimesh -c conda-forge -c nvidia -y
2. conda activate respandev3
3. pip install "scipy==1.13.1" "tensorflow<2.11" csbdeep pyqt5 "cupy-cuda11x==13.2.0" “patchify==0.2.3 <br>

Secondary environment:
1. mamba create -n respaninternal python=3.9  pytorch torchvision pytorch-cuda=12.1 scikit-image opencv -c pytorch -c nvidia -y
2. git clone -b v2.3.1 https://github.com/MIC-DKFZ/nnUNet.git
3. cd to that repo dir then pip install -e ./nnUNet
   
---
## 🛠️ Future Developments
- Our latest model uses 3D spine cores and membranes to further improve accuracy in dense environments
- Integration of Dask to remove resource limitations on processing large datasets
- Improved efficiency in batch GPU mesh measurements, neck generation, and geodesic distance measurements

---
## Benchmark: Processing and Training Times by System Configuration

| **System** | **CPU** | **RAM (GB)** | **GPU** | **Storage** | **CARE Training**<br>(10 epochs, min) | **SelfNet Training**<br>(2×10MB, 40 epochs, min) | **nnUNet (min)**<br>10MB | 100MB | 500MB | 1GB | 2.5GB | **RESPAN (min, GPU)**<br>10MB | 100MB | 500MB | 1GB | 2.5GB |
|------------|---------|--------------|---------|-------------|---------------------------------------|--------------------------------------------------|----------------------------|--------|--------|------|--------|-----------------------------|--------|--------|------|--------|
| **Mid-performance** | i9-11900K (8-core, 3.5 GHz) | 64 | RTX 3070 (8GB) | Patriot M.2 P300 1TB | 11.7 | 5 | 0.14 | 1.39 | 6.35 | 16 | 32.43 | 0.44 | 1.62 | 6.28 | 7.76 | 18.23 |
| **High-performance** | Threadripper PRO (16-core, 4.0 GHz) | 256 | RTX 4090 (24GB) | Samsung M.2 SSD 1.92TB | 3.5 | 1.5 | 0.14 | 1.39 | 6.35 | 14 | 32.43 | 0.26 | 2.33 | 8.91 | 14.07 | 26.62 |
