
### 3D CNN model

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

The recommended Python version to run this code is: Python 3.7
Libraries and packages necessary for installation are detailed in the requirements.txt file.

=========

Authors: Maxime Tchibozo, Zijing Wang

Date: Mar 1, 2021

Abstract: Implements the architecture of the 3D CNN in Keras. Included steps:

* Splits raw fMRI data into Training/Validation/Test splits and addresses data imbalance.

* Defines a 3D Residual Neural Network (ResNet) model suitable for 64 x 64 x 44 fMRI voxel data.

* Optimizes 3D ResNet hyperparameters through grid-search.

* Stores trained 3D ResNet models in ckpt format with meta files.

* Evaluates the performance of the 3D ResNet model and computes summary statistics.
