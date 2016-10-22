"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import NeuralNetwork.Network as Network
import DataExtraction.Extraction as Extraction

def test(network, images, targets):
	res = 0
	for i in range(len(images)):
		res += int(np.argmax(network.compute(images[i])) == np.argmax(targets[i]))
	return res

def dataExtraction():
	print("Download")
	fileNames= ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
	for fileName in fileNames:
		Extraction.downloadDecompress("http://yann.lecun.com/exdb/mnist/",
			fileName, "Data/")

	print("Start extraction")
	training_labels, training_images = Extraction.extractImagesLabels(
		"Data/train-labels-idx1-ubyte.gz", "Data/train-images-idx3-ubyte.gz")
	testing_labels, testing_images = Extraction.extractImagesLabels(
		"Data/t10k-labels-idx1-ubyte.gz", "Data/t10k-images-idx3-ubyte.gz")
	return training_labels, training_images, testing_labels, testing_images

def analysis(layers, learningRate, batchSize, iteration, probabilistic,
	training_labels, training_images, testing_labels, testing_images):
	print("\n" + str(layers) + " - Batch {} - Rate {} - Probabilitic {}".format(
		batchSize, learningRate, probabilistic))
	net = Network.NeuralNetwork(layers)

	print("Start backpropagation")
	net.backpropagation(training_images, training_labels, learningRate, batchSize,
		probabilistic, iteration)

	print("Test")
	print("On the training set : {} / {}".format(
		test(net, training_images, training_labels), len(training_labels)))
	print("On the testing set : {} / {}".format(
		test(net, testing_images, testing_labels), len(testing_labels)))

if __name__ == '__main__':
	trainL, trainI, testL, testI = dataExtraction()

	# Probabilistic
	analysis([784,25,10],0.001,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.005,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.05,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.1,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],1,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],1.5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],2,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],2.5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],3,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],3.5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],4,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],4.5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],5,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],5.5,10,10,True,trainL,trainI, testL, testI)

	# Non probabilistic
	analysis([784,25,10],0.001,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],0.005,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],0.05,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],0.1,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],0.5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],1,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],1.5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],2,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],2.5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],3,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],3.5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],4,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],4.5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],5,10,10,False,trainL,trainI, testL, testI)
	analysis([784,25,10],5.5,10,10,False,trainL,trainI, testL, testI)

	# Iteration
	analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,20,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,30,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,40,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,10,50,True,trainL,trainI, testL, testI)

	# Batch size
	analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,50,10,True,trainL,trainI, testL, testI)
	analysis([784,25,10],0.01,100,10,True,trainL,trainI, testL, testI)

	# Deep
	analysis([784,100,50,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,100,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,300,200,100,10],0.01,10,10,True,trainL,trainI, testL, testI)

	# High hidden links
	analysis([784,200,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,300,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,300,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,500,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,600,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,700,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,800,10],0.01,10,10,True,trainL,trainI, testL, testI)
	analysis([784,900,10],0.01,10,10,True,trainL,trainI, testL, testI)
