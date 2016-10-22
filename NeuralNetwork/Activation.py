"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

def fSigmoid(x):
	return 1. / (1. + np.exp(-x))

def dSigmoid(x):
	return fSigmoid(x) * (1. - fSigmoid(x))

def fSigUpd(x):
	return 1.7159 * np.tanh(2. / 3. * x)

def dSigUpd(x):
	return 1.7159 * 2. / 3. * (1. - np.tanh(2. / 3. * x)**2)

def fRectifier(x):
	return np.log(1. + np.exp(x))

def dRectifier(x):
	return 1. / (1. + np.exp(-x))
