# Experiment A

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

The recommended Python version to run this code is: Python 3.7
Libraries and packages necessary for installation are detailed in the requirements.txt file.

## Overview

3-class brain-state classification problem with hyperalignment and deep learning.

Donggeun Kim preprocessed data, built 1D-CNN structure in Keras and evaluated alternative benchmark models 
under guidance of Professor Xiaofu He. Maxime Tchibozo and Zijing Wang provided tremendous help in 
generating additional evaluation statistics based on further feedback from the professor. 
 
Donggeun Kim is reachable through dk2791 at columbia.edu

## Requirements and Dependencies

Code is developed in Python. We advise using Google Colab to replicate the model training steps. 

Preprocessing step is carried out using code snippets available at `Experiment_A/hyperalignment/MVA_hyperalignment.ipynb`.
To run this step, `mvpa2` package is required. `mvpa2` package was distributed in `python 2` which became 
deprecated. Input to the preprocessing step is fMRI data. Output are preprocessed voxels stored as `X_hyp_v2.csv` and 
`Y_hyp_v2.csv`. These preprocessed data can be made accessible upon requests.

Training and Evaluation step is carried out through code snippets distributed in `Experiment_A/metrics` and 
`Experiment_A/model`. If `X_hyp_v2.csv` and `Y_hyp_v2.csv` are available, then users can directly run 
`Experiment_A/main.py` to train, evaluate and plot experiment results.


## Disclaimer

Please note the scripts released in this repository are provided “as is” without warranty or guarantee of any kind, either express or implied, including but not limited to any warranty of noninfringement, merchantability, and/or fitness for a particular purpose.

The use of the scripts and the tools is At Your Own Risk, the user is responsible for reviewing before use in any environment. We are not liable for any losses, damages or other liabilities.



