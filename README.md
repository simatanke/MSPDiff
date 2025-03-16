# 【Review Under at The Visual Computer】Enhanced Passive Non-line-of-sight Imaging via Multi-Scale Polarization-Guided Diffusion Model
This repository contains the code and dataset for the paper titled ***"Enhanced Passive Non-line-of-sight Imaging via Multi-Scale Polarization-Guided Diffusion Model"***, currently under review at **The Visual Computer**. 
The dataset, named **PI-NLOS**, includes 16,000 images with polarizations at 0°, 45°, 90°, 135°, as well as polarization intensity (S0), polarization degree (DoLP), polarization angle (AoP), and ground truth images.
# PI-NLOS DataSet
* Data Collection

In an outdoor environment, we conducted data collection on the body postures and dynamic scenes of multiple models. To mitigate the impact of cold reflection effects, we set the angle between the camera and the target to **60 milliradians (mrad)**, effectively reducing interference from cold reflection halos.
Additionally, based on the focal length of the lens, we defined a region of interest (ROI) to ensure that the captured images were focused on the target area. During the data collection process, we employed a dual-person video recording approach. The specific steps are as follows:
  *  **Time Synchronization**:  At the start of image acquisition, a time anchor point was set to synchronize the initial frame positions of the non-line-of-sight (NLOS) acquisition system and the ground truth system.
  *  **Video Acquisition**: The NLOS acquisition system and the ground truth system separately captured NLOS images and ground truth images. For each action performed by the models, we ensured that they remained relatively stationary during acquisition to maintain data quality.
  *  **End Anchor Point**: At the end of video acquisition, an end anchor point was set as the termination frame to ensure data completeness and consistency.

Using this method, we collected 2,000 sets of video sequences featuring different models, postures, and scenes. Subsequently, by extracting frames from the video sequences, we obtained approximately 10,000 raw images and ground truth images.
