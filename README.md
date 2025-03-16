# 【Review Under at The Visual Computer】Enhanced Passive Non-line-of-sight Imaging via Multi-Scale Polarization-Guided Diffusion Model
This repository contains the code and dataset for the paper titled ***"Enhanced Passive Non-line-of-sight Imaging via Multi-Scale Polarization-Guided Diffusion Model"***, currently under review at **The Visual Computer**. 
The dataset, named **PI-NLOS**, includes 16,000 images with polarizations at 0°, 45°, 90°, 135°, as well as polarization intensity (S0), polarization degree (DoLP), polarization angle (AoP), and ground truth images.
# PI-NLOS DataSet
* **Data Collection**

In an outdoor environment, we conducted data collection on the body postures and dynamic scenes of multiple models. To mitigate the impact of cold reflection effects, we set the angle between the camera and the target to **60 milliradians (mrad)**, effectively reducing interference from cold reflection halos.
Additionally, based on the focal length of the lens, we defined a region of interest (ROI) to ensure that the captured images were focused on the target area. During the data collection process, we employed a dual-person video recording approach. The specific steps are as follows:
  *  **1.Time Synchronization**:  At the start of image acquisition, a time anchor point was set to synchronize the initial frame positions of the non-line-of-sight (NLOS) acquisition system and the ground truth system.
  *  **2.Video Acquisition**: The NLOS acquisition system and the ground truth system separately captured NLOS images and ground truth images. For each action performed by the models, we ensured that they remained relatively stationary during acquisition to maintain data quality.
  *  **3.End Anchor Point**: At the end of video acquisition, an end anchor point was set as the termination frame to ensure data completeness and consistency.

Using this method, we collected 2,000 sets of video sequences featuring different models, postures, and scenes. Subsequently, by extracting frames from the video sequences, we obtained approximately 10,000 raw images and ground truth images.

* **Data Registration**
  
Mechanical vibrations and spatial perspective differences cause misalignment between the four polarization angle images (0°, 45°, 90°, 135°) and the ground truth images. This misalignment prevents the images from being correctly matched, necessitating spatial and temporal registration of the processed data.
    
  *  **1. Spatial Registration**: We employ the Scale-Invariant Feature Transform (SIFT) algorithm for spatial registration. SIFT detects keypoints (local feature descriptors) in images at different scales and computes their orientations.
  *  **2. Temporal Registration** During data acquisition, all cameras were set to a frame rate of 25 fps. However, due to hardware differences and mechanical errors, the cameras operated at an average frame rate of approximately 25 fps, resulting in temporal misalignment. To correct this, we performed manual temporal registration using Adobe Premiere Pro since precise clock synchronization equipment was unavailable.

Download the dataset from the following link:
[PI-NLOS](https://github.com/Unconventional-Vision-Lab-ZZU/PI-NLOS)

After downloading the raw data, you can run the following commands to generate the S0, S1, S2, DoLP, DoP, and AoP datasets.
```
 python processData.py
```

After preparing the dataset, the folder structure should be as follows:

```
S0/
├── train/
│   ├── blur/
│   │   └── (10,000 image pairs)
│   │       └── xxxx.png
│   │       ...
│   └── sharp/
│       └── xxxx.png
│       ...
└── test/
    └── (1250 image pairs)
        └── (same as train)
```

# Dependencies
* numpy
* opencv-python
* Pillow
* pyyaml
* requests
* scikit-image
* scipy
* tb-nightly
* torch==1.9
* torchvision
* wandb
* basicsr
# train
run the command below: We  introduce the YML configuration file and explain the functionality of each parameter in our paper. The configuration file serves as a structured format for defining essential settings used in our model
```
python train.py  -opt options/train/MSPDiff.yml
```
# Test
```
python test.py  -opt options/test/MSPDiff.yml
```
# Acknowledgement
Our code is partly built upon [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks to the contributors of their great work.
