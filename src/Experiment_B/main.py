import os
import pandas as pd
import preprocessing
import train
import evaluate
import matplotlib.pyplot as plt

# Define paths
folder_data = 'data'#folder containing all the .mat files
folder_result = 'results'

# Define key parameters of the model - Training
model_type = 'resnet50_'
n_runs = 39 #number of runs that will be selected for training, validation and testing - 39 runs in total
reduce_lr_rate = 0.3
learning_rate = f'adam_reduce_lr_rate{reduce_lr_rate}'#1e-5
regularisation = 2e-4
num_epochs = 1250
nlabels = 2   #negative vs. neutral

hyperparameter_dict = dict({'model_type': model_type,'n_runs': n_runs, 'reduce_lr_rate': reduce_lr_rate,'learning_rate': learning_rate, 'learning_rate': learning_rate, 'regularisation': regularisation, 'num_epochs': num_epochs, 'nlabels': nlabels})


X_train,X_val,X_test,y_train,y_val,y_test = preprocessing.preprocess(folder_data, hyperparameter_dict)

history, path_prefix = train.training(folder_result, hyperparameter_dict, X_train, X_val, X_test, y_train, y_val, y_test)

# evaluating: load model and evaluate results
# Define key parameters of the model - Evaluation

hyperparameter_dict_eval = hyperparameter_dict #Update your hyperparameters here to evaluate a specific model's results

# Plot / Evaluate results
evaluate.roc_results(folder_result, hyperparameter_dict_eval, X_test, y_test)
evaluate.plot(folder_result, hyperparameter_dict_eval)
    






