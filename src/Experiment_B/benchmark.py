
from resnet3d import Resnet3DBuilder
import os
import tensorflow as tf
from sklearn.utils import shuffle
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier



def benchmark_helper(folder_data,n_runs):

    arr = os.listdir(folder_data)
    file_list = shuffle(arr,random_state=0)[:n_runs]
    print(len(file_list))

    print("2 classes")
    X = []
    y = []
    for root in file_list:
        data = loadmat(root)  #300, 60
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

    X_train = X_train.reshape(-1,64*64*44)
    X_val = X_val.reshape(-1,64*64*44)
    X_test = X_test.reshape(-1,64*64*44)

    return X_train,X_val,X_test,y_train,y_val,y_test

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def XG_boost(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,17)
    model_xgb = XGBClassifier(max_depth=5)
    model_xgb.fit(X_train,y_train),# eval_metric=["merror"], eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    y_pred = model_xgb.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    acc_list = []

    for i in range(10):

        a, b = unison_shuffled_copies(X_test,y_test)
        a, b = a[:len(X_test)//10,:], b[:len(X_test)//10]

        y_pred = model_xgb.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))

def unison_bootstraped_copies(a, b):
    assert len(a) == len(b)
    p = np.random.choice(list(range(len(a))), len(a), replace = True)
    return a[p], b[p]

def Randomforest(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,39)
    model_rf = RandomForestClassifier()#max_depth=5)
    model_rf = model_rf.fit(X_train,y_train)# eval_metric=["merror"], eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)


    acc_list = []

    for i in range(10):
        a, b = unison_bootstraped_copies(X_test,y_test)

        y_pred = model_rf.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))


def svm_linear(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,39)
    model_svm_linear = LinearSVC()
    model_svm_linear = model_svm_linear.fit(X_train,y_train)# eval_metric=["merror"], eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    acc_list = []

    for i in range(10):
        a, b = unison_bootstraped_copies(X_test,y_test)

        y_pred = model_svm_linear.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))


def svm_rbf(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,39)
    model_svm_rbf = SVC(kernel='rbf')
    model_svm_rbf = model_svm_rbf.fit(X_train,y_train)# eval_metric=["merror"], eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    acc_list = []

    for i in range(10):
        a, b = unison_bootstraped_copies(X_test,y_test)

        y_pred = model_svm_rbf.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))


def LDA(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,20)
    model_lda = LinearDiscriminantAnalysis()
    model_lda = model_lda.fit(X_train,y_train)

    acc_list = []

    for i in range(10):

        a, b = unison_shuffled_copies(X_test,y_test)
        a, b = a[:len(X_test)//10,:], b[:len(X_test)//10]

        y_pred = model_lda.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))


def MLP(folder_data):
    X_train,X_val,X_test,y_train,y_val,y_test = benchmark_helper(folder_data,20)
    model_mlp = MLPClassifier(max_iter=20000)
    model_mlp = model_mlp.fit(X_train,y_train)

    acc_list = []

    for i in range(10):
        a, b = unison_bootstraped_copies(X_test,y_test)

        y_pred = model_mlp.predict(a)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(b, predictions)*100
        acc_list.append(accuracy)


    print('acc:',acc_list)
    print('acc mean:',np.mean(acc_list))
    print('acc std:',np.std(acc_list))