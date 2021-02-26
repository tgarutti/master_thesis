##############################################################################
##### Get list of 10X files based on CIK and path length (For 2018 only) #####
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd
import math
loc = "/Volumes/LaCie/Data/"

def getList10X(folder_name):
    CIKs = []
    list10K = []
    list10Q = []
    for quarter in os.listdir(folder_name):
        if quarter.startswith("QTR"):
            for filename in os.listdir(folder_name+"/"+quarter):
                # Open file and load text
                file_path = folder_name+"/"+quarter+"/"+filename
                text = open(file_path,"r").read()
                
                # Read header: get company name, CIK and file type
                header = f10X.readHeader(text).tolist()
                date = "2018" + " " + quarter[0] + quarter[-1]
                if f10X.checkDates(text, 360):
                    if len(f10X.checkItems(text)) > 10:
                        CIKs.append(header[1])
                        if "10-K" == header[-1]:
                            list10K.append([date, header[1], file_path, f10X.itemize10X(file_path)])
                        elif "10-Q" == header[-1]:
                             list10Q.append([date, header[1], file_path, f10X.itemize10X(file_path)])
    CIKs = np.unique(np.array(CIKs)).tolist()
    list10X = list10K + list10Q
    return list10X, CIKs

def dataToLists():
    drive = "/Volumes/LaCie"
    CIKs = []
    for folder in reversed(os.listdir(drive)):
        if folder.startswith("10-X"):
            for year in reversed(os.listdir(drive+"/"+folder)):
                if year[0].isdigit():
                    print(year)
                    list10X, CIKtemp = getList10X(drive+"/"+folder+"/"+year)
                    f1 = drive+"/Data/"+year+"10X.pckl"
                    fd.saveFile(list10X, f1)
                    CIKs = CIKs + CIKtemp
                    del list10X
    cik_set = set(CIKs)
    CIKs = list(cik_set)
    f2 = drive+"/Data/"+"CIKs.pckl"
    fd.saveFile(CIKs, f2)
                    

def readPrices(filename, CIKs):
    prices = pd.read_csv(filename)
    prices = prices[['datadate','prccm','cik']]
    prices_final = []
    for cik in CIKs:
        for year in range(2000,2019):
            p = prices.loc[prices['cik'] == int(cik)]
            low = (year-1)*10000+1001
            high = year*10000+931
            p = p.loc[(p['datadate'] >= low) & (p['datadate'] <= high)]
            p = p[p['prccm'].notna()]
            q1 = p.loc[(p['datadate']//100%100>=10) & (p['datadate']//100%100<=12)]
            q2 = p.loc[(p['datadate']//100%100>=1) & (p['datadate']//100%100<=3)]
            q3 = p.loc[(p['datadate']//100%100>=4) & (p['datadate']//100%100<=6)]
            q4 = p.loc[(p['datadate']//100%100>=7) & (p['datadate']//100%100<=9)]
            qrt = [q1,q2,q3,q4]
            i = 1
            for q in qrt:
                a = q.empty
                if q.empty == False:
                    q = q.loc[q['datadate'].idxmax()]
                    date = str(int(q['datadate']//10000))+" Q"+str(i)
                    d={'date':date,'price':q['prccm'],'cik':cik}
                    prices_final.append(d)
                i+=1
    return pd.DataFrame(prices_final)

def constructDataset():
    for year in range(2000,2019):
        f1 = loc+str(year)+"10X.pckl"
        f2 = "/Volumes/LaCie/Data/prices.pckl"
        final10X = fd.joinPricesText(f1,f2)
        f3 = loc+str(year)+"10X_final.pckl"
        fd.saveFile(final10X, f3)
        del final10X

CIKs = fd.loadFile(loc+"CIKs.pckl")
prices = readPrices(loc+"prices.csv", CIKs)
#fd.saveFile(prices, "/Volumes/LaCie/Data/prices.pckl")
constructDataset()
                
    # prices_final = prices[~(prices['date']//100%100%3!=0)]
    # for i in range(0,len(prices)):
    #     date = prices['date'].iloc[i]
    #     yr = date//10000
    #     cik = str(prices['cik'].iloc[i])
    #     prices['cik'].iloc[i] = "0"*(10-len(cik)) + cik
    #     if date//100%100//3==4:
    #         prices['date'].iloc[i] = str(yr) + " Q1"
    #     elif date//100%100//3==1:
    #         prices['date'].iloc[i] = str(yr) + " Q2"
    #     elif date//100%100//3==2:
    #         prices['date'].iloc[i] = str(yr) + " Q3"
    #     elif date//100%100//3==3:
    #         prices['date'].iloc[i] = str(yr) + " Q4"

