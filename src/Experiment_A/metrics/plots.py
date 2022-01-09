"""
@author: Donggeun Kim, Maxime Tchibozo, and Zijing Wang
@affiliation: NYSPI, Columbia University
@date: Oct 2018 - Jan 2022
@overview: Generates Plot using evaluation statistics from `model/evaluation.py`.
"""

from model.base import retrieve_data, generate_train_test_data
from model.evaluation import compute_roc
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import os
import pandas as pd
import numpy as np
from itertools import cycle


def plot_validation_result(directory_path_to_hyperaligned_data,learning_rate=0.0045):

    savedpath = directory_path_to_hyperaligned_data

    X, y = retrieve_data(directory_path_to_hyperaligned_data)
    X_cv, _, _, _ = generate_train_test_data(X, y)

    n_leave_one_subject = len(X_cv)

    for leave_one_idx in range(n_leave_one_subject):
        print('Cross-Validation Number ', leave_one_idx, ':')
        train_result = pd.read_csv(
            os.path.join(savedpath, f'training_holdout_regularized_{leave_one_idx}_{learning_rate}.log'))


        ax = figure(0).gca()
        ax.plot('epoch', 'val_accuracy', data=train_result, color='navy')
        ax.plot('epoch', 'accuracy', data=train_result, color='crimson')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(bbox_to_anchor=(0.8, 0.09), loc='upper left', borderaxespad=0.)
        plt.title('Cross-Validation Accuracy - Subject: ' + str(leave_one_idx))
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_yticks(np.arange(0.5, 1, 0.1))
        fig = ax.get_figure()
        fig.set_figheight(9)
        fig.set_figwidth(9)
        plt.grid()
        fig.savefig(os.path.join(savedpath,"acc_cv" + str(leave_one_idx) + "_lr_" + str(learning_rate) + ".png"))

        ax2 = figure(1).gca()
        ax2.plot('epoch', 'val_loss', data=train_result, color='navy')
        ax2.plot('epoch', 'loss', data=train_result, color='crimson')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(bbox_to_anchor=(0.8, 0.95), loc='upper left', borderaxespad=0.)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
        plt.title('Cross-Validation Loss - Subject: ' + str(leave_one_idx))

        fig2 = ax2.get_figure()
        fig2.set_figheight(9)
        fig2.set_figwidth(9)
        plt.grid()
        fig2.savefig(os.path.join(savedpath,"loss_cv" + str(leave_one_idx) + "_lr_" + str(learning_rate) + ".png"))

        show()


def plot_roc(directory_path_to_hyperaligned_data,holdout_index=10,learning_rate=0.0045):
    savedpath = directory_path_to_hyperaligned_data

    fpr, tpr, roc_auc, n_classes = compute_roc(directory_path_to_hyperaligned_data, holdout_index=holdout_index)
    lw=2
    ax = figure(0).gca()
    fig = ax.get_figure()
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
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(savedpath, 'roc' + str(holdout_index) + "_lr_" + str(learning_rate) +".png"))
    show()
