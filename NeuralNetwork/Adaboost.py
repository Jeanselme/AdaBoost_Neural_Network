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

	def train(self, inputs, targets, learningRate, batchSize, maxIteration):
		"""
		Computes the combanison of weak classifiers in order to have the most
		accurate one on the given data
		"""
		inputsWeights = np.array([1/len(inputs) for i in inputs])
		for weakClassifier in self.weakClassifiers:
			print("Train weak classifier")
			error, classes = weakClassifier.backpropagationWeighted(inputs,
				np.dot(inputsWeights, len(inputsWeights)), targets, learningRate,
				batchSize, maxIteration)

			# Computes the classifier weight
			errorWellClassified = sum([np.exp(error[i]) for i in range(len(classes)) if classes[i]])
			errorMisClassified = sum([np.exp(error[i]) for i in range(len(classes)) if not classes[i]])
			alpha = 0.5 * np.log(errorWellClassified/errorMisClassified)
			self.classifiersWeights.append(alpha)

			# Computes inputs weights
			for i in range(len(classes)):
				if classes[i] :
					inputsWeights[i] *= np.exp(-alpha + error[i])
				else :
					inputsWeights[i] *= np.exp(alpha + error[i])
			np.dot(inputsWeights, 1./sum(inputsWeights))

	def compute(self, inputs):
		"""
		Computes the result respecting the different networks
		"""
		res = 0
		for i in range(len(self.weakClassifiers)):
			res += self.weakClassifiers[i].compute(inputs) *\
				self.classifiersWeights[i]
		return res
