from sklearn.utils import shuffle
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess(folder_data,n_runs):
    arr = os.listdir(folder_data)
    file_list = shuffle(arr,random_state=0)[:n_runs]
    print(len(file_list))

    print("2 classes")
    X = []
    y = []
    for root in file_list:
        mat_path = os.path.join(folder_data, root)
        data = loadmat(mat_path)  #300, 60
        non_zeros = np.argwhere(data['labels'] != [0])[:,0]
        X += list(data['I'][0][non_zeros])
        y += list(data['labels'][non_zeros].flatten())
        non_zeros = []

        non_zeros = np.argwhere(data['labels_test'] != [0])[:,0]
        X += list(data['I_test'][0][non_zeros])
        y += list(data['labels_test'][non_zeros].flatten())
        non_zeros = []

    X = np.array(X)
    y = np.array([y[i]-1 for i in range(len(y))]) #In the .mat files, there are 3 classes: 0,1,2. Here we use only classes 1 and 2 and convert them to 0,1 for the CNN to work correctly

    print('nb. samples: ',len(y))

    #save RAM
    raw_data, raw_label = [], []

    X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)
    X, y = [], []#save RAM

    X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval, test_size=0.2,stratify=y_trainval,random_state=0)
    X_trainval, y_trainval = [], [] #save RAM

    print('train bin count : ',np.bincount(y_train.astype('int')))
    print('val bin count : ',np.bincount(y_val.astype('int')))
    print('test bin count : ',np.bincount(y_test.astype('int')))

    train_bincount = np.bincount(y_train.astype('int'))

    class_percentage = [train_bincount[0]/(train_bincount[0]+train_bincount[1]),train_bincount[1]/(train_bincount[0]+train_bincount[1])]
    print('class frequency : ',class_percentage)


    y_train = tf.keras.utils.to_categorical(y_train,2)
    y_val = tf.keras.utils.to_categorical(y_val,2)
    y_test = tf.keras.utils.to_categorical(y_test,2)

    X_train = X_train.reshape(-1,64,64,44,1)
    X_val = X_val.reshape(-1,64,64,44,1)
    X_test = X_test.reshape(-1,64,64,44,1)

    print(X_train.shape)
    print(y_train.shape)

    return  X_train,X_val,X_test,y_train,y_val,y_test
