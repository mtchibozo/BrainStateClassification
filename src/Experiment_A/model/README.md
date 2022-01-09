
### 1D CNN model

Authors: Donggeun Kim, Maxime Tchibozo, Zijing Wang

Date: Oct 1, 2020

Abstract: Implements the architecture of the 1D CNN in Keras. Included steps:

* Defines a class to create models automatically with specified hyperparameters.

* Generates balanced Leave-One-Subject-Out Cross-Validation Folds using Random Oversampling.

* Iteratively trains a model on each training Fold and evaluates its performance on the test fold.

* Stores each model from each Fold in ckpt format with meta files.

* Donggeun Kim is the main contributor to contents in `base.py` and `training.py`. 
Maxime Tchibozo and Zijing Wang contributed to `evaluation.py`.
