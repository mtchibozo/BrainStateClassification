# BrainStateClassification

- An Emotional Brain State Classification pipeline using Convolutional and Residual Neural Networks.

## Overview

Code for our 2022 paper: Real-Time Emotional Brain States Classification on fMRI Data Using Deep Residual and Convolutional Networks.

## Support

For support using this code, please open an Issue on the repository, or contact mt3390@columbia.edu.

## Requirements and Dependencies

All code is developed in Python. Python versions and package dependencies are specified in the requirements.txt files located in both experiment folders.

We advise using Google Colab to replicate the model training steps.

## Usage

Steps to preprocess data, train models, evaluate their performance and compare them with classical Machine Learning algorithms are detailed in the main.py file of each Experiment.

## Input

Experiment A loads data from two CSV files titled: X_hyp_v2.csv and y_hyp_v2.csv. X_hyp_v2.csv contains an array of size n_samples × 300. y_hyp_v2.csv contains an array of size n_samples × 1.

Experiment B loads data from individual runs in Matlab (.mat) format directly. We used 39 .mat files each containing 'I', 'I_test', 'labels' and 'labels_test' entries. "I" (& "I_test") entries contain 79 or 89 64×64×44 fMRI scans and the "labels" (resp. "labels_test) entries contain the 79 or 80 integer labels associated to each fMRI scan. 

## Disclaimer

Please note the scripts released in this repository are provided “as is” without warranty or guarantee of any kind, either express or implied, including but not limited to any warranty of noninfringement, merchantability, and/or fitness for a particular purpose.

The use of the scripts and the tools is At Your Own Risk, the user is responsible for reviewing before use in any environment. We are not liable for any losses, damages or other liabilities.

## References

<a id="1">[1]</a> Tchibozo, M., Wang, Z., Kim, D. and He, X. "Emotional Brain State Classification on fMRI data using 3D Residual Neural Networks." 2020 Conference on Health Analytics, New York, USA [PDF](https://static1.squarespace.com/static/5f628d032f186434589f4eb2/t/6064f7440688f717764c4159/1617229642160/resnet-poster_3-30-2021_final.pdf) [link](https://www.dsiposters.com/posters/health-analytics-team-2)

<a id="2">[2]</a> Tchibozo, Maxime and Kim, Donggeun and Wang, Zijing and He, Xiaofu "Emotional Brain State Classification on fMRI Data Using Deep Residual and Convolutional Networks" 2022, arXiv:2210.17015 [link] 
(https://arxiv.org/abs/2210.17015)


