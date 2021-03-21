##############################################################################
##### Support Vector Machine #################################################
##############################################################################
import numpy as np
import re
import pandas as pd
import functions10X as f10X
import functionsData as fd
import funcitonsNN as fNN
import random as rd
import collections
import time
drive = '/Volumes/LaCie/Data/'

loughranDict = fd.loadFile(drive+'Loughran_McDonald_dict.pckl')
benchNNDict = fd.loadFile(drive+'dictionary_benchNN.pckl')
classNNDict = fd.loadFile(drive+'dictionary_classificationNN.pckl')
regresNNDict = fd.loadFile(drive+'dictionary_regressionNN.pckl')
dicts = [loughranDict, benchNNDict, classNNDict]

def filterDict(dictionary, percentile):
    sort = fNN.getSortedScores(dictionary)
    quant = ((sort.abs()).sort_values(ascending=False)).quantile(percentile)
    df_filtered = sort.drop(sort[(sort >= -quant) & (sort <= quant)].index)
    return df_filtered.to_dict()

benchNNDict = filterDict(benchNNDict, 0.4)
classNNDict = filterDict(classNNDict, 0.4)
regresNNDict = filterDict(regresNNDict, 0.4)