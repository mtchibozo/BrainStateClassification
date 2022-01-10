import os
import tensorflow as tf
from keras_resnet3d.resnet3d import Resnet3DBuilder

def training(folder_result, hyperparameter_dict, X_train, X_val, X_test, y_train, y_val, y_test):
    
    #Load hyperparameters
    model_type = hyperparameter_dict['model_type']
    n_runs = hyperparameter_dict['n_runs']
    learning_rate = hyperparameter_dict['learning_rate']#1e-5
    reduce_lr_rate = hyperparameter_dict['reduce_lr_rate']
    regularisation = hyperparameter_dict['regularisation']
    num_epochs = hyperparameter_dict['num_epochs']
    nlabels = hyperparameter_dict['nlabels']
    
    #Instantiate models
    path_prefix = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}')
    checkpoint_path = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}_weights.best.hdf5')
    csv_logger_path = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}.csv')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_rate,
        patience=30,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=5e-10)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
        save_weights_only=True, mode='max') # val_acc for tf.keras, val_accuracy for keras

    csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path, append=True, separator=';')

    if model_type == 'resnet101_':
        model = Resnet3DBuilder.build_resnet_101(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)


    if model_type == 'resnet50_':
        model = Resnet3DBuilder.build_resnet_50(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)


    if model_type == 'resnet34_':
        model = Resnet3DBuilder.build_resnet_34(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        batch_size=32,
                        epochs=num_epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[reduce_lr,csv_logger,checkpoint],
                        verbose=1)

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"test loss, test acc: {results}")


    return history, path_prefix