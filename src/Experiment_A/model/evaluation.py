"""
@author: Donggeun Kim, Maxime Tchibozo, Zijing Wang
@affiliation: NYSPI, Columbia University
@date: Oct 2018 - Jan 2022
@overview: Compute Evaluation Statistics
"""

import numpy as np
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
import os
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from scipy import interp
from sklearn.metrics import roc_curve
from .base import create_model, retrieve_data,generate_train_test_data,generate_cross_validated_training_data, voxel_size


def evaluate_proposed_model(directory_path_to_hyperaligned_data,learning_rate=0.0045,batch_size = 64):
    savedpath = directory_path_to_hyperaligned_data
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)

    for leave_one_idx in range(n_leave_one_subject):
        X_train, X_test, y_train_onehot, y_test_onehot = \
            generate_cross_validated_training_data(X_cv[leave_one_idx],
                                                   y_cv[leave_one_idx].reshape(-1, 1),
                                                   X_hold_out[leave_one_idx],
                                                   y_hold_out[leave_one_idx])
        model = create_model()
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        model.load_weights(
                    os.path.join(savedpath, f'weights_holdout_regularized_{leave_one_idx}_{learning_rate}.best.hdf5'))
        score, acc = model.evaluate(X_test, y_test_onehot, batch_size=batch_size, verbose=0)
        with open(os.path.join(savedpath, f"test_holdout_regularized_{leave_one_idx}_{learning_rate}.log"), 'a') as f:
            print("...Test size:" + "{:.3f}".format(X_test.shape[0]), file=f)
            print("...Test score:" + "{:.3f}".format(score), file=f)
            print("...Testing accuracy:" + "{:.3f}".format(acc), file=f)
        print(f'Test Accuracy for learning rate {learning_rate} CV index {leave_one_idx} is :', "{:.3f}".format(acc))


def compute_roc(directory_path_to_hyperaligned_data, learning_rate=0.0045, batch_size=64, holdout_index=10):

    savedpath = directory_path_to_hyperaligned_data
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)
    assert holdout_index < n_leave_one_subject, 'holdout_index should be less than number of subjects'

    X_train, X_test, y_train_onehot, y_test_onehot = \
        generate_cross_validated_training_data(X_cv[holdout_index],
                                               y_cv[holdout_index].reshape(-1, 1),
                                               X_hold_out[holdout_index],
                                               y_hold_out[holdout_index])

    model = create_model()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    model.load_weights(
        os.path.join(savedpath, f'weights_holdout_regularized_{holdout_index}_{learning_rate}.best.hdf5'))

    y_pred = model.predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    n_classes = 3

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc, n_classes


def evaluate_svc(directory_path_to_hyperaligned_data,kernel='rbf'):
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)

    for leave_one_idx in range(n_leave_one_subject):

        X_train = X_cv[leave_one_idx] # 300 refers to N_features
        Y_train = y_cv[leave_one_idx].reshape(-1,1)
        X_test = X_hold_out[leave_one_idx]
        Y_test = y_hold_out[leave_one_idx]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).reshape(-1,voxel_size,1)
        X_test = scaler.transform(X_test).reshape(-1,voxel_size,1)

        X_train, Y_train = RandomOverSampler(random_state=0).fit_resample(X_train.reshape(-1,voxel_size),Y_train.flatten())
        X_test, Y_test = RandomOverSampler(random_state=0).fit_resample(X_test.reshape(-1,voxel_size),Y_test.flatten())

        svm_rbf = SVC(kernel=kernel)
        svm_rbf.fit(X_train,Y_train)
        acc = svm_rbf.score(X_test, Y_test)
        print(f'Test Accuracy for SVC model with CV index {leave_one_idx} is :', "{:.3f}".format(acc))


def evaluate_linear_svc(directory_path_to_hyperaligned_data):
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)
    for leave_one_idx in range(n_leave_one_subject):

        X_train = X_cv[leave_one_idx] # 300 refers to N_features
        Y_train = y_cv[leave_one_idx].reshape(-1,1)
        X_test = X_hold_out[leave_one_idx]
        Y_test = y_hold_out[leave_one_idx]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).reshape(-1,voxel_size,1)
        X_test = scaler.transform(X_test).reshape(-1,voxel_size,1)

        X_train, Y_train = RandomOverSampler(random_state=0).fit_resample(X_train.reshape(-1,voxel_size),Y_train.flatten())
        X_test, Y_test = RandomOverSampler(random_state=0).fit_resample(X_test.reshape(-1,voxel_size),Y_test.flatten())

        linear_svc = LinearSVC()
        linear_svc.fit(X_train,Y_train)
        acc = linear_svc.score(X_test, Y_test)
        print(f'Test Accuracy for Linear SVC model with CV index {leave_one_idx} is :', "{:.3f}".format(acc))


def evaluate_random_forest(directory_path_to_hyperaligned_data):
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)
    for leave_one_idx in range(n_leave_one_subject):

        X_train = X_cv[leave_one_idx] # 300 refers to N_features
        Y_train = y_cv[leave_one_idx].reshape(-1,1)
        X_test = X_hold_out[leave_one_idx]
        Y_test = y_hold_out[leave_one_idx]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).reshape(-1,voxel_size,1)
        X_test = scaler.transform(X_test).reshape(-1,voxel_size,1)

        X_train, Y_train = RandomOverSampler(random_state=0).fit_resample(X_train.reshape(-1,voxel_size), Y_train.flatten())
        X_test, Y_test = RandomOverSampler(random_state=0).fit_resample(X_test.reshape(-1,voxel_size), Y_test.flatten())

        random_forest = RandomForestClassifier()
        random_forest.fit(X_train, Y_train)
        acc = random_forest.score(X_test, Y_test)
        print(f'Test Accuracy for Random Forest model with CV index {leave_one_idx} is :', "{:.3f}".format(acc))


def evaluate_hanson_neural_net(directory_path_to_hyperaligned_data,learning_rate=0.0045):
    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)
    for leave_one_idx in range(n_leave_one_subject):


        X_train, X_test, y_train, y_test = \
            generate_cross_validated_training_data(X_cv[leave_one_idx],
                                                   y_cv[leave_one_idx].reshape(-1, 1),
                                                   X_hold_out[leave_one_idx],
                                                   y_hold_out[leave_one_idx])

        model = Sequential()
        model.add(Dense(10, activation='tanh', input_shape=(voxel_size, 1)))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        batch_size = 32
        epochs = 50  # 20 epochs now

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(X_test, y_test))
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print(f'Test Accuracy for Hanson Neural Network with learning rate {learning_rate} CV index {leave_one_idx} is :', "{:.3f}".format(acc))

