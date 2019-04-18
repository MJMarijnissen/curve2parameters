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

Parameters = pd.read_csv("curvesParameters.csv", header = None)

Data, Parameters = shuffle(Data, Parameters)

DataTrain, ParametersTrain = Data[:7000], Parameters[:7000]
DataTest, ParametersTest = Data[7000:], Parameters[7000:]

model = MLPRegressor(hidden_layer_sizes=(1,100), activation='relu', max_iter=2000)

model.fit(DataTrain, ParametersTrain)

train_accuracy = model.score(DataTrain, ParametersTrain)
test_accuracy = model.score(DataTest, ParametersTest)

print(f"train accuracy:  {train_accuracy} , test accuracy:  {test_accuracy}")

#predict = model.predict(DataPredict)
#print(predict)
