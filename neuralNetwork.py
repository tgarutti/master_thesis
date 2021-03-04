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

dataset, dictionary, _, _ = fd.loadTestDataset()

def adam(grad, m, v):
    b1 = 0.9
    b2 = 0.999
    eps = 0.000000001
    rate = 0.001
    
    m_new = b1*m + (1-b1)*grad
    v_new = b2*v + (1-b2)*np.square(grad)
    
    m_hat = m_new/(1-b2)
    v_hat = v_new/(1-b2)
    
    step = (rate/(np.sqrt(v_hat)+eps)).multiply(m_hat)
    
    return step, m_new, v_new

def calculateGradients(y, y_hat, W, X):
    w0 = W[0,0]
    w1 = W[1,1]
    sumX = X.sum(axis=0)
    
    # Gradients of word values
    d1 = y.divide(y_hat)
    p1p2 = y_hat[:,0].multiply(y_hat[:,1])
    
    grad11 = -np.sum(d1.dot(w0*p1p2))
    grad12 = -np.sum(d1.multiply((-w0)*p1p2))
    grad21 = -np.sum(d1.multiply((-w1)*p1p2))
    grad22 = -np.sum(d1.multiply((w1)*p1p2))
    
    # Gradients for W
    gradw0 = np.array([-np.sum(d1.dot(sumX[0]*p1p2)),-np.sum(d1.multiply((-sumX[0])*p1p2))])
    gradw1 = np.array([-np.sum(d1.dot((-sumX[1])*p1p2)),-np.sum(d1.multiply(sumX[0]*p1p2))])
    
    grad = np.array([grad11, grad12, grad21, grad22, gradw0, gradw1])
    
    return grad

def backPropagation(y, y_hat, W, X, m, v):
    grad = calculateGradients(y, y_hat, W, X)
    step, m, v = adam(grad, m, v)
    return

def calculateLoss(y, y_hat):
    loss = np.multiply(y, np.log(y_hat))
    return np.sum(loss)
    
def softmax(A):
    e = np.exp(A)
    return e / np.sum(e, axis=0, keepdims=True)

def forwardPropagation(batch, batch_dict):
    W = np.array([[0.0001,0],
                  [0,0.0001]])
    y = []
    y_hat = []
    X = []
    for item in batch:
        text = item[-1]
        price_boolean = item[3]
        price = item[4]
        y.append(price_boolean)
        text = f10X.cleanText(text)
        
        # From text to values
        values = [batch_dict[w][0:2] for w in text if w in batch_dict]
        values = np.array(values)
        X.append(values)
        
        # Summation layer - results in 2x1 vector
        sumValues = (values.sum(axis=0))
        
        # First linear layer - results in 2x1 vector
        linlayer = W.dot(sumValues)
        
        # Softmax
        y_hat.append(softmax(linlayer))
    y = np.column_stack(y)
    y_hat = np.column_stack(y_hat)
    return y, y_hat, X

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
    w1 = 1
    w2 = 1
    W = np.array([[w1,0],
                  [0,w2]])
    return W

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