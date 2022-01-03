from resnet3d import Resnet3DBuilder
import matplotlib.pyplot as plt
import numpy as np
import os

#import tensorflow.keras as keras

def roc_results(folder_result,n_runs,X_test,y_test):

    model_type = 'resnet50_'

    reduce_lr_rate = 0.3
    learning_rate = 'adam_reduce_lr_rate'+str(reduce_lr_rate)#1e-5
    num_epochs = 1250

    for regularisation in [2e-4]:

        load_model_path = os.path.join(folder_result , model_type , 'first_',str(n_runs),'runs_lr_',str(learning_rate) , '_reg', str(regularisation) , '_num_epochs',str(num_epochs) , '_weights.best.hdf5')

        if model_type == 'resnet101_':
            model = Resnet3DBuilder.build_resnet_101(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)

        if model_type == 'resnet50_':
            model = Resnet3DBuilder.build_resnet_50(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)

        if model_type == 'resnet34_':
            model = Resnet3DBuilder.build_resnet_34(input_shape=(64, 64, 44, 1), num_outputs=2, reg_factor=regularisation)

        model.compile(optimizer="adam",
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.load_weights(load_model_path)

        print("Evaluate on test data")
        results = model.evaluate(X_test, y_test, batch_size=32)
        print("test loss, test acc:", results)
        predictions = model.predict(X_test)
        print(predictions)

        Y_test_onehot = y_test.copy()
        y_proba_pred = np.array(predictions)

        from sklearn.metrics import roc_curve, auc
        from scipy import interp
        from itertools import cycle

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
        plt.title('Receiver operating characteristic - reg=0.0002')
        plt.legend(loc="lower right")
        plt.savefig(folder_result+'roc_reg0.0002.png')
        plt.show()