# Just Do It
```
git clone https://github.com/wangguan1995/3D_ESRGAN_Turbulence.git
cd 3D_ESRGAN_Turbulence
checkout master

wget https://dataset.bj.bcebos.com/PaddleScience/cylinder3D/3D_ESRGAN/data.zip
unzip data.zip

pip install -r requirements.txt     # install libs

python nor_fluc3d.py                 # normalize data
python ESRGAN_3D.py                  # train and plot png
```

# Paper
2022 Three-dimensional ESRGAN for super-resolution reconstruction of turbulent flows with tricubic interpolationbased transfer learning

# Code from:
https://fluids.pusan.ac.kr/ fluids/65416/subview.do

# Colorful pictures
![image](https://github.com/user-attachments/assets/1bd928c0-13ab-4ac7-9adf-9917c9c38803)

![image](https://github.com/user-attachments/assets/423d611f-d16a-4a14-9595-5bef3aef2e2a)

![image](https://github.com/user-attachments/assets/d0a3ba40-a882-46e6-aa5a-2f7ca44dfd8a)


# Overview
3D-ESRGAN is developed to reconstruct 3D super-resolution channel flow from low-resolution data.

# Data (Very Poor)
100 snapshots (channel flow at Rer =180) and its low-resolution data are provided for tutorial. 
Before the input, the data should be normalized using code nor_fluc3d.py to get normalized data.

# Training
Use 3d-esrgan,py to train the deep learning model. It will save architecture and weights files automatically.
