# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:25 2019

SK-learn regressor for guessing the parameters of a curve, based on the curve's shape

@author: Kubus
"""

from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv("curves.csv", header = None)
Data.drop(columns=101, inplace=True)
Data.drop(index=0, inplace=True)
#Data = Data.T

Parameters = pd.read_csv("curvesParameters.csv", header = None)
#Parameters = Parameters.T

Data, Parameters = shuffle(Data, Parameters)

DataTrain, ParametersTrain = Data[:7000], Parameters[:7000]
DataTest, ParametersTest = Data[7000:], Parameters[7000:]

model = MLPRegressor(hidden_layer_sizes=(1,100), 
                     activation='relu', 
                     max_iter=20000, 
                     learning_rate_init = 0.00001)

model.fit(DataTrain, ParametersTrain)

train_accuracy = model.score(DataTrain, ParametersTrain)
test_accuracy = model.score(DataTest, ParametersTest)

print(f"train accuracy:  {train_accuracy} , test accuracy:  {test_accuracy}")

from prep_experm_data import read_data, generate_function

DataEXP = read_data()
#print(DataPredict[1])
predict = model.predict(DataEXP.iloc[[1]])
print(predict)

x=generate_function(predict[0][0], predict[0][1],predict[0][2])

plt.figure(1)
plt.plot(x, label="predicted")
plt.plot(DataEXP.iloc[1], label="experiment")
plt.legend(loc='best')
plt.show()

"""TO DO: 
    -Metric measuring the distance between the 2 curves
    -Automated sampling of experimental data""" 