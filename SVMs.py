##############################################################################
##### Support Vector Machine #################################################
##############################################################################
import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import functionsSVM as fSVM
import random as rd
import collections
import time
from sklearn import svm

drive = '/Volumes/LaCie/Data/'

loughranDict = fd.loadFile(drive+'Loughran_McDonald_dict.pckl')
benchNNDict = fd.loadFile(drive+'dictionary_benchNN.pckl')
classNNDict = fd.loadFile(drive+'dictionary_classificationNN.pckl')
regresNNDict = fd.loadFile(drive+'dictionary_regressionNN.pckl')
dictionaries = [benchNNDict, classNNDict, regresNNDict]

dictionaries = fSVM.filterDicts(loughranDict, dictionaries, 0.4)
dict_names = ['Loughran', 'Benchmark', 'Classification', 'Regression']

def SVMDataset(dictionaries, dict_names):
    train, test = dict(),dict()
    for d in dict_names:
        train[d] = []
        test[d] = []
    for year in range(2000,2015):
        print(year)
        filename = drive+str(year)+'10X_final.pckl'
        X = fSVM.getScores(filename, dictionaries, dict_names)
        for d in dict_names:
            train[d].extend(X[d])
    for year in range(2015,2019):
        print(year)
        filename = drive+str(year)+'10X_final.pckl'
        X = fSVM.getScores(filename, dictionaries, dict_names)
        for d in dict_names:
            test[d].extend(X[d])
    for name in dict_names:
        train[name] = np.row_stack(train[name])
        test[name] = np.row_stack(test[name])
    return train, test
train = fd.loadFile(drive+'train.pckl')
test = fd.loadFile( drive+'test.pckl')
y = 0

start = time.time()
#def runSVM(train, test, dict_names, y):
results = dict()
for name in dict_names:
    results[name] = []
for name in dict_names:
    X_train = train[name][:,0:11]
    X_train = fSVM.cleanMat(X_train)
    X_train = fSVM.normalizeX(X_train)
    y_train = train[name][:,11:]
    y_train = y_train[:,y]
    
    X_test = test[name][:,0:11]
    X_test = fSVM.cleanMat(X_test)
    X_test = fSVM.normalizeX(X_test)
    y_test = test[name][:,11:]
    y_test = y_test[:,y]
    
    X_train = X_train[:40000, 4:]
    y_train = y_train[:40000]
    X_test = X_test[:10000, 4:]
    y_test = y_test[:10000]
    
    X_train = np.delete(X_train, 4, 1)
    X_test = np.delete(X_test, 4, 1)
    clf = svm.NuSVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(np.sum(np.equal(y_pred,y_test)))
    
    results[name].extend([y_pred, y_test])
#train, test = SVMDataset(dictionaries, dict_names)
#y_results = runSVM(train, test, dict_names, 0)
end = time.time()
print(end-start)