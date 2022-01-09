"""
@author: Donggeun Kim
@affiliation: NYSPI, Columbia University
@date: Oct 2018 - Oct 2020
@overview: Generates Cross-Subject-Validation Fold datasets using ANOVA and MVPA hyperalignment.
@input: Preprocessed voxels post Anova and Hyperlaignment
@output: Estimated Model and Training Information
"""

from tensorflow import keras
import glob
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from .base import create_model, retrieve_data, generate_train_test_data, generate_cross_validated_training_data


def train_model(directory_path_to_hyperaligned_data, learning_rate=0.0045, epochs=5000,batch_size = 64):

    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, y_cv, X_hold_out, y_hold_out = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)

    for leave_one_idx in range(n_leave_one_subject):

        X_train, X_test, y_train_onehot, y_test_onehot = \
            generate_cross_validated_training_data(X_cv[leave_one_idx],
                                                   y_cv[leave_one_idx].reshape(-1, 1),
                                                   X_hold_out[leave_one_idx],
                                                   y_hold_out[leave_one_idx])

        print("to create")
        model = create_model()

        savedpath = directory_path_to_hyperaligned_data


        # If the model weights have not yet been computed (i.e model did not exist), create weights file, otherwise, load the weights. Files are located in the plateau folder
        best_model_path = os.path.join(savedpath, f"weights_holdout_regularized_{leave_one_idx}_{learning_rate}.best.hdf5")
        print(best_model_path)
        if len(glob.glob(
                os.path.join(savedpath, f"weights_holdout_regularized_{leave_one_idx}_{learning_rate}.best.hdf5"))) == 0:
            print("created a new model")
        else:
            print("loaded weights from file")
            model.load_weights(
                os.path.join(savedpath, f'weights_holdout_regularized_{leave_one_idx}_{learning_rate}.best.hdf5'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])


        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50,
                                      min_lr=0.00001)  # Reduce learning rate when a metric has stopped improving.
        checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', verbose=0, save_best_only=True,
                                     mode='max')
        csv_logger = keras.callbacks.CSVLogger(
            os.path.join(savedpath, f'training_holdout_regularized_{leave_one_idx}_{learning_rate}.log'))
        print("Before Fit")
        model.fit(X_train, y_train_onehot,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, y_test_onehot),
                  callbacks=[csv_logger, checkpoint, reduce_lr])
        print(model.summary())









