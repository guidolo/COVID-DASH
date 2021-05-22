# model equations for the scaled SIR model for python 2.7
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

import numpy as np


import numpy as np

def model(ini, time_step, params):
	Y = np.zeros(3) #column vector for the state variables
	X = ini
	mu = 0
	beta = params[0]
	gamma = params[1]

	Y[0] = mu - beta*X[0]*X[1] - mu*X[0] #S
	Y[1] = beta*X[0]*X[1] - gamma*X[1] - mu*X[1] #I
	Y[2] = gamma*X[1] - mu*X[2] #R

	return Y

def x0fcn(params, data):
	S0 = 1.0 - (data[0]/params[2])
	I0 = data[0]/params[2]
	R0 = 0.0
	X0 = [S0, I0, R0]

	return X0

def yfcn(res, params):
	return res[:,1]*params[2]

def model2(ini, t, params):
    S, I, R = ini
    beta = params[0]
    gamma = params[1]
    N = params[2]
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I
    return (dSdt, dIdt, dRdt)

def x0fcn2(params, data):
    S0 = 1.0 - (data[0]/params[2])
    I0 = data[0]/params[2]
    R0 = 0.0
    return (S0, I0, R0)

def ini2(N, I0, R0):
    I0 = I0
    R0 = R0
    S0 = N - I0 - R0
    return (S0, I0, R0)

def yfcn2(res, params):
	return res[:,1]*params[2]
