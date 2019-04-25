# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:46:43 2019

@author: Kubus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    """Normalizes the X values by dividing the first column by max value"""
    max = data[0].max()
    data[0]=np.divide(data[0],max)
    return data

#def strip(data):
#    """Picks out only 100 data points from the given set"""
#    lenght = len(data)
#    step = round(lenght/100)
#    ret=pd.DataFrame()
#    for i in range(0,lenght,step):
#        ret.append(data[lenght*i])
def read_data():
    exp = pd.read_csv("exp0_test0.csv", header = None)
    exp = exp.T
    exp = normalize(exp)
    return exp
        

def generate_function(K,eps,n):
    x = np.arange(0,1.01,0.01)
    return K*(eps + x)**n
#plt.plot(exp[1])
#plt.show()
