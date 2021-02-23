##############################################################################
##### Get list of 10X files based on CIK and path length (For 2018 only) #####
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import random as rd
import collections

dataset, dictionary = fd.loadTestDataset()

def batchDictionary(batch):
    batch_dict = {}
    counter_dict = {}
    i=0
    for doc in batch:
        for k, value in collections.Counter(f10X.cleanText(doc[-1])).items():
            if k in dictionary:
                if k in batch_dict:
                    vec = np.zeros((len(batch),1))
                    vec[tuple([i,0])] = value
                    counter_dict[k] +=  vec
                else:
                    vec = np.zeros((len(batch),1))
                    vec[tuple([i,0])] = value
                    batch_dict.update({k: dictionary[k][0:2]})
                    counter_dict.update({k: vec})
        d = {k: dictionary[k][0:2] for k in f10X.cleanText(doc[-1]) if k in dictionary}
        batch_dict.update(d)
        i+=1
    return batch_dict, pd.DataFrame(counter_dict)

def batchIndices(batch, batch_dict):
    names = pd.DataFrame(batch_dict).columns()

def newEpoch():
    index = 0
    stop = False
    return index, stop

def initializeCoefficients():
    w_1 = 1
    w_2 = 1
    return w_1, w_2

def nextBatch(index):
    batch_size,_ = setHyperparameters()
    if len(dataset) - batch_size < 20:
        batch = dataset[index:]
        stop = True
    batch = dataset[index:(index+batch_size)]
    index = index+batch_size
    return batch, index
    
def setHyperparameters():
    batch_size = 20
    epochs = 10
    return batch_size, epochs