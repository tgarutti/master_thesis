##############################################################################
##### Get list of 10X files based on CIK and path length (For 2018 only) #####
##############################################################################
import os
import numpy as np
import pandas as pd
import functions10X as f10X
import functionsData as fd

# Location external drive
folder_name = "/Users/user/Documents/Erasmus/QFMaster/Master Thesis/Python_code/2018"
list201810K = []
list201810Q = []
list201810K_full = []
list201810Q_full = []

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
                    if "10-K" == header[-1]:
                        list201810K.append(np.array([date, header[1], file_path]))
                        list201810K_full.append([date, header[1], file_path, f10X.itemize10X(file_path)])
                    elif "10-Q" == header[-1]:
                         list201810Q.append(np.array([date, header[1], file_path]))
                         list201810Q_full.append([date, header[1], file_path, f10X.itemize10X(file_path)])

## Sort list/arrays of 10Ks and 10Qs by the CIK
array10K = np.vstack(list201810K)
array10K = array10K[array10K[:,2].argsort()]
array10Q = np.vstack(list201810Q)
array10Q = array10Q[array10Q[:,2].argsort()]
CIKs = np.unique(array10Q[:,2])

array10K_final = []
array10Q_final = []
cik_final = []
yrs_min = 1

for i in CIKs:
    temp10Q = array10Q[array10Q[:,2]==i,:]
    temp10K = array10K[array10K[:,2]==i,:]
    qrt, indQ = np.unique(temp10Q[:,0], return_index=True)
    yr, indK = np.unique(temp10K[:,0], return_index=True)
    if qrt.shape[0] >= yrs_min*3 and yr.shape[0] >= yrs_min:
        array10Q_final.append(temp10Q[indQ,:])
        array10K_final.append(temp10K[indK,:])
        cik_final.append(i)
        
array10Q_final = np.vstack(array10Q_final)
array10K_final = np.vstack(array10K_final)
list201810K = array10K_final.tolist()
list201810Q = array10Q_final.tolist()
CIKs = np.unique(array10Q_final[:,2]).tolist()
fileNames10K = array10K_final[:,4].tolist()
fileNames10Q = array10Q_final[:,4].tolist()
fd.saveFile(list201810K, "10Ks_2018.pckl")
fd.saveFile(list201810Q, "10Qs_2018.pckl")
fd.saveFile(CIKs, "CIKs_2018.pckl")
fd.saveFile(fileNames10Q, "fileNames10K_2018.pckl")
fd.saveFile(fileNames10K, "fileNames10Q_2018.pckl")

fd.saveFile(list201810K_full, "10Ks_2018TEXT.pckl")
fd.saveFile(list201810Q_full, "10Qs_2018TEXT.pckl")
fd.listToText(CIKs, "CIKs_2018.txt")


# Clean up variables
del array10K, array10K_final, array10Q, array10Q_final,\
    cik_final, date, file_path, filename, folder_name, header, i, indK, indQ,\
    qrt, quarter, temp10K, temp10Q, text, yr, yrs_min
