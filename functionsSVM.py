##############################################################################
##### Functions for SVMs #####################################################
##############################################################################
import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import functionsNN as fNN
import random as rd
import collections
import time

def percentileDict(dictionary, percentile):
    sort = fNN.getSortedScores(dictionary)
    quant = ((sort.abs()).sort_values(ascending=False)).quantile(percentile)
    df_filtered = sort.drop(sort[(sort >= -quant) & (sort <= quant)].index)
    return df_filtered.to_dict()

def filterDicts(loughranDict, dictionaries, percentile):
    for i in range(len(dictionaries)):
        dictionaries[i] = percentileDict(dictionaries[i], percentile)
    Ldf = pd.DataFrame(loughranDict)
    loughranDict = (Ldf.loc['Positive']-Ldf.loc['Negative']).to_dict()
    dictionaries.insert(0, loughranDict)
    return dictionaries

def nPos(A):
    array = A.copy()
    array[array<0] = 0
    array[array>0] = 1
    return int(sum(array))

def nNeg(A):
    array = A.copy()
    array[array>0] = 0
    array[array<0] = 1
    return int(sum(array))

def getOmega(text, dictionary):
    text = f10X.cleanText(text)
    values = [dictionary[w] for w in text if w in dictionary]
    if len(values)==0:
        return []
    else:  
        len_text = len(text)
        n_sentWords = len(values)
        values = np.array(values)
        n_pos = nPos(values)
        n_neg = nNeg(values)
        score = int(sum(values))
        return [len_text, n_sentWords, n_pos, n_neg, score]

def getScores(filename, dictionaries, dict_names):
    dataset = fd.loadFile(filename)
    data = dict()
    for d in dict_names:
        data[d] = []
    for item in dataset:
        if item[6][0] >=0:
            y = [item[4][0],item[5],item[6][0],1]
        else:
            y = [item[4][0],item[5],item[6][0],0]
        x1 = np.concatenate(np.row_stack(item[7])[:,0:2]).tolist()
        n = 6-len(x1)
        for i in range(n):
            x1.insert(0, 0)
        for i in range(len(dictionaries)):
            d = dictionaries[i]
            d_name = dict_names[i]
            x2 = getOmega(item[-1], d)
            if len(x2) > 0:
                x = np.array(x1+x2+y)
                data[d_name].append(x)
    return data