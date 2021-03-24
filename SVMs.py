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
    return train, test
start = time.time()
train, test = SVMDataset(dictionaries, dict_names)
end = time.time()
print(end-start)