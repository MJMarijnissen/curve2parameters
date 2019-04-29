# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:50:13 2019

This file generates the required function for the train/test set

@author: Kubus
"""

import numpy as np
from prep_experm_data import generate_function
import csv

K = np.arange(1,10)
eps = np.arange(1,10)
n = np.arange(1,10)

f=[]
for k in K:
    for epsilon in eps:
        for nico in n:
            f.append(generate_function(k,epsilon, nico))

with open("function_train_test.csv", mode="w") as function_train_test:
    function_train_test = csv.writer(function_train_test)
    for i in f:
        function_train_test.writerow(i)
    
#function_train_test.close()