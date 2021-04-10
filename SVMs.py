##############################################################################
##### Support Vector Machines #################################################
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
from sklearn import metrics

drive = '/Volumes/LaCie/Data/'
def svmDictionaries():
    loughranDict = fd.loadFile(drive+'Loughran_McDonald_dict.pckl')
    benchNNDict = fd.loadFile(drive+'dictionary_benchNN.pckl')
    classNNDict = fd.loadFile(drive+'dictionary_classificationNN.pckl')
    regresNNDict = fd.loadFile(drive+'dictionary_regressionNN.pckl')
    dictionaries = [benchNNDict, classNNDict, regresNNDict]
    
    dictionaries = fSVM.filterDicts(loughranDict, dictionaries, 0.4)
    return dictionaries

dictionaries = fd.loadFile(drive+'SVM_dictionaries.pckl')
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

train = fd.loadFile(drive+'train_final.pckl')
test = fd.loadFile( drive+'test_final.pckl')

start = time.time()
def runSVM(train, test, dict_names, ker, yValues):
    results = dict()
    for name in dict_names:
        results[name] = []
    for name in dict_names:
       for y in yValues: 
            X_train = train[name][:,2:13]
            X_train = fSVM.cleanMat(X_train.astype(np.float))
            X_train = fSVM.normalizeX(X_train)
            y_train = train[name][:,13:]
            y_train = y_train[:,y]
            
            X_test = test[name][:,2:13]
            X_test = fSVM.cleanMat(X_test.astype(np.float))
            X_test = fSVM.normalizeX(X_test)
            y_test = test[name][:,13:]
            y_test = y_test[:,y]
            
            X_train = X_train[:10000, 4:]
            y_train = y_train[:10000]
            X_test = X_test[:2000, 4:]
            y_test = y_test[:2000]
            
            X_train = np.delete(X_train, 4, 1)
            X_test = np.delete(X_test, 4, 1)
            X_train = np.delete(X_train, 1, 1)
            X_test = np.delete(X_test, 1, 1)
            
            X_train = fSVM.filterX(X_train)
            X_test = fSVM.filterX(X_test)
            if y==1 or y==2:
                clf = svm.NuSVR(gamma='auto', kernel = 'linear')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[name].append([y_pred, y_test])
            elif y==0 or y==3:
                clf = svm.NuSVC(gamma='auto', nu=0.5, kernel = ker)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[name].append([y_pred, y_test])
    return results

def resultsForecasts(forecasts, dict_names, yValues):
    results={}
    for y in yValues:
        res = []
        for name in dict_names:
            if y==0 or y==3:
                yPred = forecasts[name][y][0].astype(np.int)
                yReal = forecasts[name][y][1].astype(np.int)
                n = sum(np.equal(yPred,yReal))
                p = n/len(yReal)
                p_pos = sum(np.equal(yPred[yReal==1],yReal[yReal==1]))/len(yReal[yReal==1])
                p_neg = sum(np.equal(yPred[yReal==0],yReal[yReal==0]))/len(yReal[yReal==0])
                print(metrics.accuracy_score(yReal, yPred))
                res.append([n,p,p_pos,p_neg])
            elif y==1 or y==2:
                yPred = forecasts[name][y][0].astype(float)
                yReal = forecasts[name][y][1].astype(float)
                mse = np.sqrt((np.square(yPred - yReal)).mean())
                res.append(mse)
        res = np.row_stack(res)
        resDF = pd.DataFrame(res)
        resDF.index = dict_names
        if len(res[0,:])==4:
            colNames = ['Number', 'Percent', 'Percent (Pos.)', 'Percent (Neg.)']
            resDF.columns = colNames
        else:
            colNames = ['MSE']
            resDF.columns = colNames
        results[y] = resDF
    return results
        
#train, test = SVMDataset(dictionaries, dict_names)
forecasts = []
#forecasts = runSVM(train, test, dict_names, 'rbf', [0,1,2,3])
#forecasts = fd.loadFile(drive+'forecasts_temp.pckl')
#results = resultsForecasts(forecasts, dict_names, [0,1,2,3])

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuning_parameters = {'kernel': ['rbf','linear','poly2','poly3','sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 'auto', 'scale'], 'C': [1,10,100,1000]}
gridSearch = fSVM.manualGridSearch(train, test, dict_names, 3, [2000,5000,10000,20000], [400,1000,2000,4000], tuning_parameters)
end = time.time()
print(end-start)