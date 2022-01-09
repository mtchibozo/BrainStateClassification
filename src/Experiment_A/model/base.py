"""
@author: Donggeun Kim
@affiliation: NYSPI, Columbia University
@date: Oct 2018 - Jan 2022
@overview: Create Proposed Model Structure in Keras and Generate Training and Test data
"""

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv1D
import os
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler


bias_regularization = 0.005
kernel_regularization = 0.005
voxel_size = 300


def create_model():
    # Model structure built by Donggeun Kim under guidance of Professor Xiaofu He
    model = Sequential()
    model.add(Conv1D(128, (10), input_shape=(300,1)))
    model.add(Conv1D(128, (10)))
    model.add(Conv1D(128, (10)))
    model.add(Conv1D(64, (10)))
    model.add(Conv1D(64, (10)))
    model.add(Conv1D(64, (10)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
                           bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
                       bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
                    bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
                bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
              bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
          bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(16, activation='relu',kernel_regularizer=keras.regularizers.l1(kernel_regularization),
      bias_regularizer=keras.regularizers.l1(bias_regularization)))
    model.add(Dense(3, activation='softmax'))
    return model


def retrieve_data(directory_path_to_hyperaligned_data):
    # Hard-coded file name to avoid confusion when users replicate the research
    # The file name is identical to files generated from ipython notebooks in hyperalignment folder
    y_path = os.path.join(directory_path_to_hyperaligned_data, 'Y_hyp_v2.csv')
    X_path = os.path.join(directory_path_to_hyperaligned_data, 'X_hyp_v2.csv')
    y = np.loadtxt(y_path, delimiter=",")
    X = np.loadtxt(X_path, delimiter=",")
    return X, y


def generate_train_test_data(X, y):
    tmp = list(range(0, 4800, 400))
    X_hold_out = []  # Holdout = testing | Max
    X_cv = []
    y_hold_out = []
    y_cv = []
    n_leave_one_subject = len(tmp) - 1
    for i in range(n_leave_one_subject):
        X_hold_out += [X[tmp[i]:tmp[i + 1]]]
        y_hold_out += [y[tmp[i]:tmp[i + 1]]]
        X_cv += [np.concatenate((X[0:tmp[i]], X[tmp[i + 1]:]))]
        y_cv += [np.concatenate((y[0:tmp[i]], y[tmp[i + 1]:]))]
    return X_cv, y_cv, X_hold_out, y_hold_out


def generate_cross_validated_training_data(X_cv, y_cv,X_hold_out,y_hold_out,output_onehot=True):
    X_train = X_cv  # 300 refers to N_features
    y_train = y_cv.reshape(-1, 1)
    X_test = X_hold_out
    y_test = y_hold_out

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).reshape(-1, voxel_size, 1)
    X_test = scaler.transform(X_test).reshape(-1, voxel_size, 1)

    # Y_..._onehot contains one hot encoded Y_... i.e, if Y[i] = 0, Y_onehot[i] = [1,0,0]


    X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train.reshape(-1, voxel_size), y_train.flatten())
    X_train = X_train.reshape(-1, voxel_size, 1)
    X_test, y_test = RandomOverSampler(random_state=0).fit_resample(X_test.reshape(-1, voxel_size), y_test.flatten())
    X_test = X_test.reshape(-1, voxel_size, 1)

    if output_onehot == True:
        y_train= np.eye(3)[y_train.astype(int)].reshape(-1, 3)
        y_test = np.eye(3)[y_test.astype(int)].reshape(-1, 3)
    return X_train, X_test, y_train, y_test
