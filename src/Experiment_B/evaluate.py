import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from keras_resnet3d.resnet3d import Resnet3DBuilder

#import tensorflow.keras as keras

def roc_results(folder_result, hyperparameter_dict, X_test, y_test):
    #Load hyperparameters
    model_type = hyperparameter_dict['model_type']
    n_runs = hyperparameter_dict['n_runs']
    learning_rate = hyperparameter_dict['learning_rate']#1e-5
    num_epochs = hyperparameter_dict['num_epochs']
    regularisation = hyperparameter_dict['regularisation']
    nlabels = hyperparameter_dict['nlabels']
    
    #Load model
    load_model_path = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}_weights.best.hdf5')

    if model_type == 'resnet101_':
        model = Resnet3DBuilder.build_resnet_101(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)

    if model_type == 'resnet50_':
        model = Resnet3DBuilder.build_resnet_50(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)

    if model_type == 'resnet34_':
        model = Resnet3DBuilder.build_resnet_34(input_shape=(64, 64, 44, 1), num_outputs=nlabels, reg_factor=regularisation)

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(load_model_path)

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"test loss, test acc:{results}")
    predictions = model.predict(X_test)
    print(predictions)

    Y_test_onehot = y_test.copy()
    y_proba_pred = np.array(predictions)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(Y_test_onehot[:, i], y_proba_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_onehot.ravel(),y_proba_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(12,9))
    n_classes=2
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

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic - reg={regularisation}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(folder_result, f'roc_reg{regularisation}.png'))
    plt.show()

def plot(folder_result, hyperparameter_dict):
    
    #Load hyperparameters
    model_type = hyperparameter_dict['model_type']
    n_runs = hyperparameter_dict['n_runs']
    learning_rate = hyperparameter_dict['learning_rate']#1e-5
    regularisation = hyperparameter_dict['regularisation']
    num_epochs = hyperparameter_dict['num_epochs']

    #Load results
    path_prefix = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}')
    csv_logger_path = os.path.join(folder_result, f'{model_type}first_{n_runs}runs_lr_{learning_rate}_reg{regularisation}_num_epochs{num_epochs}.csv')

    resnet_df = pd.read_csv(csv_logger_path,sep=";")
    resnet_df = resnet_df[resnet_df.epoch > 200]
    resnet_df['epoch'] = resnet_df['epoch'].apply(lambda x: x-200)

    plt.figure(figsize=(12,9))
    plt.plot(resnet_df["epoch"],resnet_df["loss"],color='navy',label='loss')
    plt.plot(resnet_df["epoch"],resnet_df["val_loss"],color='darkred',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.title(f'Loss and val loss as functions of epoch - 3D ResNet50 - (reg: {regularisation}')
    #plt.yticks(np.arange(0.3,1,0.1))
    plt.savefig(f'{path_prefix}_loss' +'.png')
    plt.show()

    plt.figure(figsize=(12,9))
    plt.plot(resnet_df["epoch"],resnet_df["val_acc"],color='darkred',label='val_accuracy')
    plt.plot(resnet_df["epoch"],resnet_df["acc"],color='navy',label='accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.title(f'Accuracy and val accuracy as functions of epoch - 3D ResNet50 - (reg: {regularisation}')
    plt.savefig(f'{path_prefix}_acc' +'.png')
    plt.show()

