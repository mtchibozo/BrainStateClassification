from resnet3d import Resnet3DBuilder
import os
import tensorflow as tf

def training(folder_result,n_runs,X_train,X_val,X_test,y_train,y_val,y_test):

    model_type = 'resnet50_'

    reduce_lr_rate = 0.3
    learning_rate = 'adam_reduce_lr_rate'+str(reduce_lr_rate)#1e-5
    num_epochs = 1000

    for regularisation in [5e-3]:
        path_prefix = folder_result + model_type +'first_'+str(n_runs)+'runs_lr_'+str(learning_rate) + '_reg'+ str(regularisation) + '_num_epochs'+str(num_epochs)
        checkpoint_path = os.path.join(folder_result , 'saved_models/' ,  model_type ,'first_',str(n_runs),'runs_lr_',str(learning_rate) , '_reg', str(regularisation) , '_num_epochs',str(num_epochs) , '_weights.best.hdf5')
        csv_logger_path = os.path.join(path_prefix,'.csv')

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
            model = Resnet3DBuilder.build_resnet_101(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)


        if model_type == 'resnet50_':
            model = Resnet3DBuilder.build_resnet_50(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)


        if model_type == 'resnet34_':
            model = Resnet3DBuilder.build_resnet_34(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)

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
        print("test loss, test acc:", results)

        return history,path_prefix