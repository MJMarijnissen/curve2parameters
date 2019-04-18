# curve2parameters
Neural network that uses a provided curve to guess the parametrs that were used to create the curve

curves.csv contains the curves that were created with the parameters in curvesParameters.csv

Normaly, the curves originate from experimental data and the curve is fitted with parameters guessed by the user. The idea here is to train a nearal network with pre-generated curves (for given parametrs), so that it can provide an accurate guess of parametrs when it is provided with experimental data. 

Current highest test accuracy: 
sklearn MLPRegressor:  0.9998769403115264
