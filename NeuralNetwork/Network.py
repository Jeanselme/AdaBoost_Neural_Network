"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from NeuralNetwork.Activation import fSigmoid as fActivation, dSigmoid as dActivation
from NeuralNetwork.Cost import fQuadratic as fCost, dQuadratic as dCost

class NeuralNetwork:
	"""
	Class of the neural network which works with backpropagation
	"""

	def __init__(self, dims):
		"""
		Creates a neural network respecting the different given dimensions,
		this should be a list of number, wher the first represents the number of
		inputs and the last, the number of outputs.
		The neural network will be fully connected
		"""
		self.layersNumber = len(dims) - 1
		self.weights = []
		self.biases = []
		np.random.seed(42)
		for d in range(self.layersNumber):
			self.weights.append(np.random.randn(dims[d+1], dims[d]))
			self.biases.append(np.random.randn(dims[d+1], 1))

	def compute(self, inputs):
		"""
		Computes the result of the netword by propagation
		"""
		res = inputs
		for layer in range(self.layersNumber):
			weight = self.weights[layer]
			bias = self.biases[layer]
			res = fActivation(np.dot(weight, res) + bias)
		return res

	def backpropagation(self, inputs, targets, learningRate, batchSize,
		probabilistic, maxIteration):
		"""
		Computes the backpropagation of the gradient in order to reduce the
		quadratic error
		"""
		error, pastError = 0, 0
		for iteration in range(maxIteration):
			# Decrease the learningRate
			if iteration > 1 and error > pastError :
				learningRate /= 2
			pastError = error

			# Changes order of the dataset
			if probabilistic :
				permut = np.random.permutation(len(targets))
				inputs = inputs[permut]
				targets = targets[permut]

			# Computes each image
			for batch in range(len(targets)//batchSize - 1):
				totalDiffWeight = [np.zeros(weight.shape) for weight in self.weights]
				totalDiffBias = [np.zeros(bias.shape) for bias in self.biases]

				# Computes the difference for each batch
				for i in range(batch*batchSize,(batch+1)*batchSize):
					diffWeight, diffBias, diffError = self.computeDiff(inputs[i], targets[i])
					totalDiffWeight = [totalDiffWeight[j] + diffWeight[j]
										for j in range(len(totalDiffWeight))]
					totalDiffBias = [totalDiffBias[j] + diffBias[j]
										for j in range(len(totalDiffBias))]
					error += diffError

				# Update weights and biases of each neuron
				self.weights = [self.weights[i] - learningRate*totalDiffWeight[i]
									for i in range(len(totalDiffWeight))]
				self.biases = [self.biases[i] - learningRate*totalDiffBias[i]
									for i in range(len(totalDiffBias))]
			print("{} / {}".format(iteration+1, maxIteration), end = '\r')
		print("\nBackPropagation done !")

	def computeDiff(self, input, target):
		"""
		Executes the forward and backward propagation for the given data
		"""
		diffWeight = [np.zeros(weight.shape) for weight in self.weights]
		diffBias = [np.zeros(bias.shape) for bias in self.biases]

		# Forward
		# layerSum contents all the result of nodes
		# layerAct = fActivation(layerSum)
		layerSum = []
		lastRes = input
		layerAct = [lastRes]
		for layer in range(self.layersNumber):
			layerRes = np.dot(self.weights[layer], lastRes) + self.biases[layer]
			lastRes = fActivation(layerRes)
			layerSum.append(layerRes)
			layerAct.append(lastRes)

		# Backward
		diffError = sum(fCost(lastRes, target))
		delta = dCost(lastRes, target) * dActivation(lastRes)
		diffBias[-1] = delta
		diffWeight[-1] = np.dot(delta, layerAct[-2].transpose())
		for layer in reversed(range(self.layersNumber-1)):
			delta = np.dot(self.weights[layer+1].transpose(), delta) *\
				dActivation(layerSum[layer])
			diffBias[layer] = delta
			diffWeight[layer] = np.dot(delta, layerAct[layer].transpose())

		return diffWeight, diffBias, diffError
