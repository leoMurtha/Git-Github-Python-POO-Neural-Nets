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
		for k in range(1, self.len):
			self.layers[k].z = np.dot(self.layers[k].weights, self.layers[k - 1].outputs)
			#print("<{}> * <{}> + <{}> = {}\n\n".format(np.shape(self.layers[k].weights), np.shape(self.layers[k - 1].outputs) , np.shape(np.transpose(self.layers[k].biases)), np.shape(self.layers[k].z)))
			self.layers[k].z = np.add(self.layers[k].z, np.transpose(self.layers[k].biases))
			#print("<{}> * <{}> + <{}> = {}\n\n".format(np.shape(self.layers[k].weights), np.shape(self.layers[k - 1].outputs) , np.shape(np.transpose(self.layers[k].biases)), np.shape(self.layers[k].z)))
	
			self.layers[k].outputs = sigmoid(self.layers[k].z)
			#print((self.layers[k].z), self.layers[k].outputs)
			
	# Returns a tuple (gctw, gctb) wich are
	# The gradients of cost function with respect to weight and biases
	# gctw and gctb are numpy arrays
	def backpropagation(self, output):
		gctw = [np.zeros((self.layers[i].weights.shape)) for i in range(1, self.len)]
		gctb = [np.zeros((self.layers[i].biases.shape)) for i in range(1, self.len)]
		
		# Backward pass, delta from last layer
		# Calculates partial derivative of the cost with respect to the input from layer L
		delta = np.multiply(mse_derivative(self.layers[-1].outputs, output), sigmoid_derivative(self.layers[-1].z))
		#print(np.shape(mse_derivative(self.layers[-1].outputs, output)), np.shape(sigmoid_derivative(self.layers[-1].z)))
		gctb[-1] = delta
		gctw[-1] = np.multiply(self.layers[-2].outputs, delta)
		
		# Putting the back in the backpropagation
		for l in reversed(range(1, self.len - 1)):
			#print(l, l+1)
			# Calculates partial derivative of the cost with respect to the input from layer l
			
			#print(np.shape(np.transpose(self.layers[l+1].weights)), np.shape(np.transpose(sigmoid_derivative(self.layers[l].z))), np.shape(delta))
			#print(np.shape(np.dot(np.transpose((self.layers[l+1].weights)), delta)))
			delta = np.multiply(np.dot(np.transpose((self.layers[l+1].weights)), delta), sigmoid_derivative(self.layers[l].z))
			gctb[l-1] = delta
			gctw[l-1] = np.dot((self.layers[l-1].outputs), np.transpose(delta))			

		#print("<><><>")

		return (gctw, gctb)

	def gradientdescent(self, gctw, gctb, lr):
		for l in range(1, self.len):
			#print("Weightss:{}".format(self.layers[l].weights))
			#print("Gweights:{}".format(np.transpose(gctw[l-1])))

			self.layers[l].weights = self.layers[l].weights - lr*(np.transpose(gctw[l-1]))
			self.layers[l].biases = self.layers[l].biases - lr*(np.transpose(gctb[l-1]))
			
	# Trains the NN
	# Computes the backpropagation every single input (SLOW)
	# Using stochastic gradient descent (Every input we update)
	def train(self, training_input, training_output, lr=0.1, epochs=10):
		for i in range(epochs):
			cost = 0
			for _input, _output in zip(training_input, training_output):
				_input = np.reshape(_input, (len(_input), 1))
				self.feedforward(_input)
				cost += mse(self.layers[-1].outputs, _output)/len(training_input)
				gctw, gctb = self.backpropagation(_output)
				self.gradientdescent(gctw, gctb, lr)
				
			#print("Cost for epoch[{}] = {}".format(i, cost))

	# Returns an np.array with the predictions given a test_set
	def predict(self, test_set):

		for i in range(len(test_set)):
			print(i)
			self.feedforward(np.reshape(test_set[i], (len(test_set[i]), 1)))
			print(self.layers[-1].outputs)
