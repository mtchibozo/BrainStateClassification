
### Hyperalignment

Author: Donggeun Kim

Date: Oct 1, 2020

Abstract: Generates Hyperaligned voxel vectors (PyMVPA) from raw 3D fMRI data (Matlab). Included steps:

* Applies a brain tissue mask to perform a skull extraction of the brain tissue voxels from each 3D fMRI image.

* Builds initial pre-ANOVA, pre-hyperalignment datasets for each Leave-One-Subject-Out Cross-Validation Fold.

* Applies ANOVA to training dataset voxels and retrieves the most predictive voxels for each Fold.

* Builds a hyperalignment transformation matrix using training ANOVA-selected voxels for each Fold.

* Projects training and validation voxels into a common representational space using hyperalignment for each Fold.

* Stores the  300-voxel hyperaligned training and test vectors for each fold in as NumPy arrays with their labels in X_hyp_v2.csv and Y_hyp_v2.csv
