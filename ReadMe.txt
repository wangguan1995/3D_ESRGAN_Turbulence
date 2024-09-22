-Overview-
3D-ESRGAN is developed to reconstruct 3D super-resolution channel flow from low-resolution data.

-Dependencies-
1. Python 3.6-3.8
2. tensorflow >=2.2.0<2.4.0 (cuDNN=7.6, CUDA=10.1 for tensorflow-gpu)
3. Numpy <1.19

-Data preparation-
100 snapshots (channel flow at Rer =180) and its low-resolution data are provided for tutorial. Before the input, the data should be normalized using
code nor_fluc3d.py to get normalized data.

-Training- 
Use 3d-esrgan,py to train the deep learning model. It will save architecture and weights files automatically.