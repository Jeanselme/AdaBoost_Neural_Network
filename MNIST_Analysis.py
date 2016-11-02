"""
	Adaboost Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import NeuralNetwork.Adaboost as Adaboost
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

def analysis(layers, training_labels, training_images, testing_labels, testing_images,
	 learningRate, batchSize, iteration):
	ada = Adaboost.AdaBoost()

	for layer in layers:
		ada.addWeakClassifier(layer)

	print("Start training")
	ada.train(training_images, training_labels, learningRate, batchSize,
		iteration)

	print("Test")
	print("On the training set : {} / {}".format(
		test(ada, training_images, training_labels), len(training_labels)))
	print("On the testing set : {} / {}".format(
		test(ada, testing_images, testing_labels), len(testing_labels)))

if __name__ == '__main__':
	trainL, trainI, testL, testI = dataExtraction()

	analysis([[784,25,10],[784,25,10],[784,25,10],[784,25,10],[784,25,10],
		[784,25,10],[784,25,10],[784,25,10],[784,25,10],[784,25,10]],
		trainL, trainI, testL, testI, 0.01, 10, 3)

	analysis([[784,25,10],[784,25,10],[784,25,10]],trainL, trainI, testL, testI,
		0.01, 10, 10)

	analysis([[784,25,10],[784,25,10],[784,25,10],[784,25,10],[784,25,10]],
		trainL, trainI, testL, testI, 0.01, 10, 10)
