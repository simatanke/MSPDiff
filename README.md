# MSPDiff
MSPDiff employs a progressive training schedule, transitioning from lower to higher resolutions of polarized LWIR images, to guide the iterative gradient descent. 

# Dependencies
* Python 3.9
* Pytorch (2.1.0) - Different versions may cause some errors.
* scikit-image
* opencv-python
* Tensorboard
* timm
* einops
* numpy

# Dataset
The polarization dataset is collected by our laboratory and includes Stokes parameters (S0, S1, S2), Degree of Linear Polarization (DoLP), and Angle of Polarization (AoP). 
the original dataset is available at https://github.com/Unconventional-Vision-Lab-ZZU/PI-NLOS

# Train
### python train.py --model_name "MSPDiff" --mode "train" --data_dir "dataset/Polarization_Dataset"

# Test
### python test.py --model_name "MSPDiff" --mode "train" --data_dir "dataset/Polarization_Dataset"
