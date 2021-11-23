
### 3D CNN model

Authors: Maxime Tchibozo

Date: Mar 1, 2021

Abstract: Implements the architecture of the 3D CNN in Keras. Included steps:

* Splits raw fMRI data into Training/Validation/Test splits and addresses data imbalance.

* Defines a 3D Residual Neural Network (ResNet) model suitable for 64 x 64 x 44 fMRI voxel data.

* Optimizes 3D ResNet hyperparameters through grid-search.

* Stores trained 3D ResNet models in ckpt format with meta files.

* Evaluates the performance of the 3D ResNet model and computes summary statistics.
