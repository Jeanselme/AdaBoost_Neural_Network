"""
	Adaboost Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
import NeuralNetwork.Network as Network

class AdaBoostNeuralNetwork:
	"""
	Class of multiple weak neural networks
	"""

	def __init__(self):
		"""
		Creates several neural networks as asked in the rnnLayers
		"""
		self.weakClassifiers = []
		self.classifiersWeights = []

	def addWeakClassifier(self, layers):
		"""
		Adds a neural network in the Adaboost
		"""
		self.weakClassifiers.append(Network.NeuralNetwork(layers))

	def train(self, inputs, targets, learningRate, batchSize,
		probabilistic, maxIteration):
		"""
		Computes the combanison of weak classifiers in order to have the most
		accurate one on the given data
		"""
		# TODO : Update the weight with respect of the error
		inputsWeights = np.array([1/len(inputs) for i in inputs])
		for weakClassifier in self.weakClassifiers:
			print("Train weak classifier")
			weakClassifier.backpropagationWeighted(inputs, inputsWeights, targets,
				learningRate, batchSize, probabilistic, maxIteration)
			self.classifiersWeights.append(1./len(self.weakClassifiers))

	def compute(self, inputs):
		"""
		Computes the result respecting the different networks
		"""
		res = 0
		for i in range(len(self.weakClassifiers)):
			res += self.weakClassifiers[i].compute(inputs) *\
				self.classifiersWeights[i]
		return res
