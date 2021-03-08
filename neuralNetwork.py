##############################################################################
##### Neural Network #########################################################
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import random as rd
import collections
import time

dataset, dictionary, _, _ = fd.loadTestDataset()
start = time.time()

def update(batch_dictDF, W, stepP, stepN, stepW, m, v):
    batch_dictDF.loc['pos'] = batch_dictDF.loc['pos']-stepP
    batch_dictDF.loc['neg'] = batch_dictDF.loc['neg']-stepN
    batch_dictDF.loc['mp'] = m[0]
    batch_dictDF.loc['mn'] = m[1]
    batch_dictDF.loc['vp'] = v[0]
    batch_dictDF.loc['vn'] = v[1]
    W = W - np.diag(stepW)
    return batch_dictDF, W

def update2(batch_dictDF, W, stepP, stepN, stepW, m, v):
    dfP = batch_dictDF.loc['pos']-stepP
    dfN = batch_dictDF.loc['neg']-stepN
    dfP[dfP < 0] = 0
    dfN[dfN > 0] = 0
    batch_dictDF.loc['pos'] = dfP
    batch_dictDF.loc['neg'] = dfN
    batch_dictDF.loc['mp'] = m[0]
    batch_dictDF.loc['mn'] = m[1]
    batch_dictDF.loc['vp'] = v[0]
    batch_dictDF.loc['vn'] = v[1]
    W = W - np.diag(stepW)
    return batch_dictDF, W

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

##### DIVIDE BY GRADIENTS BY LENGTH OF BATCH
def calculateGradients(y, y_hat, W, X, N):
    batch_len=len(y[0,:])
    if N == 0:
        w0 = W[0,0]
        w1 = W[1,1]
    else:
        w0 = W[0,0]/N
        w1 = W[1,1]/N
    sumX = X.sum(axis=0)
    d1 = np.divide(y,y_hat)
    p1p2 = np.multiply(y_hat[0,:],y_hat[1,:])
    
    # Gradients of word values
    derP = np.multiply(np.array([w0*p1p2,(-w0)*p1p2]),d1)
    derN = np.multiply(np.array([(-w1)*p1p2,w1*p1p2]),d1)
    gradP = -derP.sum(axis=(0))
    gradN = -derN.sum(axis=(0))    
    
    # Gradients for W
    der0 = np.multiply(np.array([np.multiply(X[:,0],p1p2),np.multiply((-X[:,0]),p1p2)]),d1)
    der1 = np.multiply(np.array([np.multiply((-X[:,1]),p1p2),np.multiply(X[:,1],p1p2)]),d1)
    gradw0 = -der0.sum(axis=(0,1))/batch_len
    gradw1 = -der1.sum(axis=(0,1))/batch_len
    
    grad = np.array([gradP, gradN, gradw0, gradw1])
    
    return grad

def backPropagation(batch_dictDF, batch_mat, y, y_hat, W, X, m, v, N):
    grad = calculateGradients(y, y_hat, W, X, N)
    gradP = (grad[0]*batch_mat.T).sum(1)/len(batch_mat.columns)
    gradN = (grad[1]*batch_mat.T).sum(1)/len(batch_mat.columns)
    gradW = np.array([grad[2], grad[3]])
    stepP, m[0], v[0] = adam(gradP, m[0], v[0])
    stepN, m[1], v[1] = adam(gradN, m[1], v[1])
    stepW, m[2], v[2] = adam(gradW, m[2], v[2])
    batch_dictDF, W = update(batch_dictDF, W, stepP, stepN, stepW, m, v)
    return batch_dictDF, W, m[2], v[2]

def calculateLoss(y, y_hat):
    loss = np.multiply(y, np.log(y_hat))
    return np.sum(loss)
    
def softmax(A):
    e = np.exp(A)
    return e / np.sum(e, axis=0, keepdims=True)

def forwardPropagation(batch, batch_dict, batch_mat, W, N):
    y = []
    y_hat = []
    X = []
    i = 0
    for item in batch:
        i+=1
        text = item[-1]
        price_boolean = item[3]
        price = item[4]
        y.append(price_boolean)
        text = list(set(f10X.cleanText(text)))        
        # From text to values
        doc = "doc"+str(i)
        values = [[batch_mat.at[doc,w]*batch_dict[w]['pos'],batch_mat.at[doc,w]*batch_dict[w]['neg']] for w in text if w in batch_dict]
        #values = [[batch_dict[w]['pos'],batch_dict[w]['neg']] for w in text if w in batch_dict]
        values = np.array(values)
                
        # Summation layer - results in 2x1 vector
        sumValues = (values.sum(axis=0))#/N
        X.append(sumValues)
        
        # First linear layer - results in 2x1 vector
        linlayer = W.dot(sumValues)
        
        # Softmax
        y_hat.append(softmax(linlayer))
    y = np.column_stack(y)
    y_hat = np.column_stack(y_hat)
    X = np.row_stack(X)
    return y, y_hat, X

def euclideanNorm(A):
    A = np.square(A)
    colSum = A.sum(0)
    return np.sqrt(colSum)
    
def batchDictionary(batch):
    fullTexts = ''
    fullTexts = ''.join([fullTexts + doc[-1] for doc in batch])
    inter = list(set(f10X.cleanText(fullTexts)).intersection(dictionary.keys()))
    colNames = inter
    rowNames = ["doc"+str(i+1) for i in range(len(batch))]
    zeros = np.zeros((len(rowNames),len(colNames)))
    batch_mat = pd.DataFrame(zeros, index=rowNames, columns=colNames)
    batch_dict = {}
    i=1
    for doc in batch:
        docList = list(set(f10X.cleanText(doc[-1])).intersection(inter))
        d = {}
        d = {k: dictionary[k] for k in docList}
        freq = collections.Counter(f10X.cleanText(doc[-1]))
        for k in docList:
            rowStr = "doc"+str(i)
            batch_mat.loc[rowStr,k] = freq[k]
        batch_dict.update(d)
        i+=1
    return batch_dict, batch_mat

def newEpoch():
    index = 0
    stop = False
    return index, stop

def initializeX():
    for key in dictionary.keys():
        dictionary[key]['pos'] = rd.randint(-50, 50)/100
        dictionary[key]['neg'] = rd.randint(-50, 50)/100

def initializeCoefficients():
    w1 = 1
    w2 = 1
    W = np.array([[w1,0],
                  [0,w2]])
    m = np.array([0,0])
    v = np.array([0,0])
    return W, m, v

def nextBatch(dataset, index, batch_size, stop):
    if len(dataset) - batch_size < batch_size:
        batch = dataset[index:]
        stop = True
    else:
        batch = dataset[index:(index+batch_size)]
    index = index+batch_size
    return batch, index, stop
    
def setHyperparameters():
    batch_size = 40
    epochs = 2
    return batch_size, epochs

#def runNeuralNetwork():
#Initialize Neural Network
batch_size, epochs = setHyperparameters()
W, m_coef, v_coef = initializeCoefficients()
initializeX()
loss = []
for j in range(epochs):
    i, stop = newEpoch()
    while stop == False:
        start2 = time.time()
        batch, i, stop = nextBatch(dataset, i, batch_size, stop)
        batch_dict, batch_mat = batchDictionary(batch)
        euclid = euclideanNorm(batch_mat)
        batch_mat1 = batch_mat/euclid
        N = 0
        #N = (batch_mat.sum(1)).mean()
        #batch_mat1 = batch_mat/N
        batch_dictDF = pd.DataFrame(batch_dict)
        m = [batch_dictDF.loc['mp'],batch_dictDF.loc['mn'], m_coef]
        v = [batch_dictDF.loc['vp'],batch_dictDF.loc['vn'], v_coef]
        y, y_hat, X = forwardPropagation(batch, batch_dict, batch_mat1, W, N)
        loss.append(calculateLoss(y, y_hat))
        batch_dictDF, W, m_coef, v_coef = backPropagation(batch_dictDF, batch_mat, y, y_hat, W, X, m, v, N)
        d = batch_dictDF.to_dict()
        dictionary.update(d)
        end2 = time.time()
        print(end2-start2)
#    return loss, W

#loss, W = runNeuralNetwork()

# initializeX()
# batch_size,_ = setHyperparameters()
# batch,_,_ = nextBatch(dataset, 0, batch_size, False)
# batch_dict, batch_mat = batchDictionary(batch)
# #euclid = euclideanNorm(batch_mat)
# N = (batch_mat.sum(1)).mean()
# #batch_mat1 = batch_mat/euclid
# batch_mat2 = batch_mat/N
# batch_dictDF = pd.DataFrame(batch_dict)
# m = [batch_dictDF.loc['mp'],batch_dictDF.loc['mn']]
# v = [batch_dictDF.loc['vp'],batch_dictDF.loc['vn']]
# W, m_coef, v_coef = initializeCoefficients()
# y, y_hat, X = forwardPropagation(batch, batch_dict, batch_mat2, W, N)
# grad = calculateGradients(y, y_hat, W, X, N)
# gradP = (grad[0]*batch_mat.T).sum(1)/batch_size
# gradN = (grad[1]*batch_mat.T).sum(1)/batch_size
# gradW = np.array([grad[2], grad[3]])
# stepP, m[0], v[0] = adam(gradP, m[0], v[0])
# stepN, m[1], v[1] = adam(gradN, m[1], v[1])
# stepW, m_coef, v_coef = adam(gradW, m_coef, v_coef)
# stepW = np.diag(stepW)
# W = W-stepW
# batch_dictDF.loc['pos'] = batch_dictDF.loc['pos']-stepP
# batch_dictDF.loc['neg'] = batch_dictDF.loc['neg']-stepN
# batch_dictDF.loc['mp'] = m[0]
# batch_dictDF.loc['mn'] = m[1]
# batch_dictDF.loc['vp'] = v[0]
# batch_dictDF.loc['vn'] = v[1]
# d = batch_dictDF.to_dict()
# dictionary.update(d)
end = time.time()
print(end-start)







        