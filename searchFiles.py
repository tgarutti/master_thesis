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
loc = "/Volumes/LaCie/Data/"
search = []
for year in range(2000,2019):
        print(year)
        f1 = loc+str(year)+"10X_final.pckl"
        dataset = fd.loadFile(f1)
        for item in dataset:
            wc = f10X.wordCount(item[-1])
            search.append([item[0], wc])