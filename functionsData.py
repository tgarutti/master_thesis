##############################################################################
##### Functions for data loading/saving ######################################
##############################################################################       
import pickle
import pandas as pd
import numpy as np
import math

## Pickle: save and load file
def saveFile(file, filename):
    f = open(filename, 'wb')
    pickle.dump(file, f, protocol=-1)
    f.close()
    
def loadFile(filename):
    f = open(filename, 'rb')
    file = pickle.load(f)
    f.close()
    return file

## Write a list to txt file 
def listToText(listObj, filename):
    string = ''
    for i in listObj:
        string = string + str(i) + "\n"
    text = open(filename, 'w')
    text.write(string)
    text.close()
    
def readPrices(filename):
    prices = pd.read_csv(filename)
    prices = prices[~(prices['date']//100%100%3!=0)]
    for i in range(0,len(prices)):
        date = prices['date'].iloc[i]
        yr = date//10000
        cik = str(prices['cik'].iloc[i])
        prices['cik'].iloc[i] = "0"*(10-len(cik)) + cik
        if date//100%100//3==4:
            prices['date'].iloc[i] = str(yr) + " Q1"
        elif date//100%100//3==1:
            prices['date'].iloc[i] = str(yr) + " Q2"
        elif date//100%100//3==2:
            prices['date'].iloc[i] = str(yr) + " Q3"
        elif date//100%100//3==3:
            prices['date'].iloc[i] = str(yr) + " Q4"
    return prices.to_numpy()

#def loadDataset():
    
def joinPricesText(f1, f2):
    text10X = loadFile(f1)
    prices = loadFile(f2)
    prices = prices.to_numpy()
    CIKs = np.unique(prices[:,2])
    text10Xfinal = []
    for item in text10X:
        date = item[0]
        date2 = incrementQuarter(date, 1)
        cik = item[1]
        if cik in CIKs:
            p_temp = prices[prices[:,2]==cik]
            if date in p_temp[:,0] and date2 in p_temp[:,0]:
                p1 = p_temp[p_temp[:,0] == date][0,1]
                p2 = p_temp[p_temp[:,0] == date2][0,1]
                if p1!=0:
                    p_change = (p2-p1)/p1
                    y = np.array([1,0])
                    y = np.array([0,1]) if p_change < 0 else y
                    item.insert(-1, y)
                    item.insert(-1, p_change)
                    text10Xfinal.append(item)
    return text10Xfinal

def loadTestDataset():
    dataset = loadFile("dataset2018.pckl")
    dictionary = loadFile("dict2018.pckl")
    return dataset, dictionary

def incrementQuarter(date, increment):
    [yr, qrt] = date.split(" Q")
    yr = int(yr)
    qrt = int(qrt)
    if increment > 0:
        if qrt < 4:
            qrt = qrt+1
        else:
            yr = yr+1
            qrt = 1
    if increment < 0:
        if qrt > 1:
            qrt = qrt-1
        else:
            yr = yr-1
            qrt = 4
    return str(yr) + " Q" + str(qrt)