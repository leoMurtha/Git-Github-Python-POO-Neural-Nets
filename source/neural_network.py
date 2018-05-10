import numpy as np
from layer import *
from function import *


class NeuralNetwork(object):
	"""A neural network composed of layers"""

	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.layers = [Layer()]
		self.len = 1

	# Adds a layer to the NN
	def add(self, layer):
		self.layers.append(layer)
		self.len += 1

	def feedforward(self, input):
		self.layers[0] = Layer(num_of_neurons=len(input), outputs=input)
		for k in xrange(1, self.len):
			self.layers[k].z = np.dot(self.layers[k].weights, self.layers[k - 1].outputs) + np.transpose(self.layers[k].biases)
			self.layers[k].outputs = sigmoid(self.layers[k].z)
	
	# Returns a tuple (gctw, gctb) wich are
	# The gradients of cost function with respect to weight and biases
	# gctw and gctb are numpy arrays
	def backpropagation(self, output):
		gctw = [np.zeros((self.layers[i].weights.shape)) for i in xrange(1, self.len)]
		gctb = [np.zeros((self.layers[i].biases.shape)) for i in xrange(1, self.len)]

		# Backward pass, delta from last layer
		delta = mse(self.layers[-1].outputs, output) * sigmoid_derivative(self.layers[-1].z)
		gctb[-1] = delta
		gctw[-1] = np.dot(delta, np.transpose(self.layers[-2].outputs))		

		# Putting the back in the backpropagation
		for l in reversed(xrange(1, self.len - 1)):
			delta = np.dot((np.transpose(self.layers[l+1].weights) * delta), sigmoid_derivative(self.layers[l].z))
			gctb[l-1] = delta
			gctw[l-1] = np.dot(self.layers[l].outputs, delta)			

		return (gctw, gctb)

	def gradientdescent(self, gctw, gctb, lr):
		for l in xrange(1, self.len):
			#print("{} WEIFHTS {}\n\n".format(l, gctw[l]))
			self.layers[l].weights = self.layers[l].weights - lr*gctw[l-1]
			#print("<<{} @@@ {}>>".format(, lr*gctb[l-1]))
			self.layers[l].biases = self.layers[l].biases - lr*gctb[l-1]

	# Trains the NN
	# Computes the backpropagation every single input (SLOW)
	# Next update : batches
	def train(self, training_input, training_output, lr=0.1, epochs=10, iterations=100):
		for i in xrange(epochs):
			for _input, _output in zip(training_input, training_output):
				self.feedforward(_input)
				gctw, gctb = self.backpropagation(_output)
				self.gradientdescent(gctw, gctb, lr)
			print("Cost for epoch[{}] = {}".format(i, mse(self.layers[-1].outputs, _output)))