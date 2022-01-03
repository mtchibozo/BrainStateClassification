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

import pandas as pd
hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = path_prefix + '_history.json'

with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# save to json:
hist_json_file = os.path.join(path_prefix + '_history.json')

with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# evaluating: load model for ROC resnet50_first_39runs_lr_adam_reduce_lr_rate0.3_reg0.0005_num_epochs1250_weights.best

evaluate.roc_results(folder_result,n_runs,X_test,y_test)




