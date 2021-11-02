
### 1D CNN model - Benchmarks, Plots and Evaluation

Authors: Maxime Tchibozo, Donggeun Kim

Date: Oct 1, 2020

Abstract: Evaluates the performance of the 1D CNN. Included steps:

* Loads trained 1D CNN models from each fold and evaluates their ROC curves, precision, recall and confusion matrices.

* Loads the hyperaligned LOOCV datasets and trains classical algorithms (MLP, XGBoost, SVM, ...) for benchmarking.

* Loads the saved histories of each LOOCV model and plots accuracy and loss as functions of epoch.
