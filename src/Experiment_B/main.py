import os
import pandas as pd
import preprocessing
import train
import evaluate
import matplotlib.pyplot as plt


folder_root = '3D_CNN_Cross-Subject'#Update to your main folder, this folder must contain a data and a result subfolder
folder_data = os.path.join(folder_root, 'data/')#folder containing all the .mat files
folder_result = os.path.join(folder_root, 'result/','2021-models-results/')



#define key parameters of the model
n_runs = 39 #number of runs that will be selected for training, validation and testing - 39 runs in total
nlabels = 2   #negative vs. neutral

X_train,X_val,X_test,y_train,y_val,y_test = preprocessing.preprocess(folder_data,n_runs)

history,path_prefix = train.training(folder_result,n_runs,X_train,X_val,X_test,y_train,y_val,y_test)

# evaluating: load model for ROC resnet50_first_39runs_lr_adam_reduce_lr_rate0.3_reg0.0005_num_epochs1250_weights.best

evaluate.roc_results(folder_result,n_runs,X_test,y_test)


## plotting the best model

model_type = 'resnet50_'

reduce_lr_rate = 0.3
learning_rate = 'adam_reduce_lr_rate'+str(reduce_lr_rate)#1e-5
num_epochs = 1250

for regularisation in [2e-4]:#, 9e-4, 1.25e-3, 2.5e-3, 5e-3]:
    path_prefix = folder_result + model_type +'first_'+str(n_runs)+'runs_lr_'+str(learning_rate) + '_reg'+ str(regularisation) + '_num_epochs'+str(num_epochs)

    checkpoint_path = path_prefix + '_weights.best.hdf5'
    csv_logger_path = os.path.join(path_prefix,'.csv')

import pandas as pd

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
plt.title('Loss and val loss as functions of epoch - 3D ResNet50 - (reg: '+str(regularisation)+')')
#plt.yticks(np.arange(0.3,1,0.1))
#plt.savefig(path_prefix + '_loss' +'.png')
plt.show()

plt.figure(figsize=(12,9))
plt.plot(resnet_df["epoch"],resnet_df["val_acc"],color='darkred',label='val_accuracy')
plt.plot(resnet_df["epoch"],resnet_df["acc"],color='navy',label='accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid()
plt.title('Accuracy and val accuracy as functions of epoch - 3D ResNet50 - (reg: ' + str(regularisation) + ')')
#plt.savefig(path_prefix + '_acc' +'.png')
plt.show()






