##############################################################################
##### Functions for Neural Network ###########################################
##############################################################################
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import random as rd
import collections
import time

def euclideanNorm(A):
    A = np.square(A)
    colSum = A.sum(0)
    return np.sqrt(colSum)

def newEpoch():
    index = 0
    stop = False
    return index, stop

def initializeX(dictionary):
    for key in dictionary.keys():
        dictionary[key]['pos'] = rd.randint(-50, 50)/100
        dictionary[key]['neg'] = rd.randint(-50, 50)/100
    return dictionary

def crossEntropyLoss(y, y_hat):
    loss = np.multiply(y, np.log(y_hat))
    return -np.sum(loss)/len(y[0,:])

def softmax(A):
    e = np.exp(A)
    return e / np.sum(e, axis=0, keepdims=True)

def nextBatch(dataset, index, batch_size, stop):
    if len(dataset) -index - batch_size < batch_size:
        batch = dataset[index:]
        stop = True
    else:
        batch = dataset[index:(index+batch_size)]
    index = index+batch_size
    return batch, index, stop

def adam(grad, m, v):
    b1 = 0.9
    b2 = 0.999
    eps = 0.000000001
    rate = 0.001
    
    m_new = b1*m + (1-b1)*grad
    v_new = b2*v + (1-b2)*np.square(grad)
    
    m_hat = m_new/(1-b2)
    v_hat = v_new/(1-b2)
    
    step = np.multiply((rate/(np.sqrt(v_hat)+eps)),m_hat)
    
    return step, m_new, v_new

def getSortedScores(dictionary):
    df = pd.DataFrame(dictionary)
    dfScore = df.loc['pos']-df.loc['neg']
    dfScore = dfScore.sort_values(ascending=False)
    return dfScore