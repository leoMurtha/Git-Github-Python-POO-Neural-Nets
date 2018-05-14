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
			self.layers[k].z = np.dot(self.layers[k].weights, np.transpose(self.layers[k - 1].outputs))
			self.layers[k].z = np.add(self.layers[k].z, self.layers[k].biases)

			self.layers[k].outputs = (self.layers[k].z)
			#print("<{}> * <{}> + <{}> = {}\n\n".format(self.layers[k].weights, self.layers[k - 1].outputs , np.transpose(self.layers[k].biases), self.layers[k].z))
	
	# Returns a tuple (gctw, gctb) wich are
	# The gradients of cost function with respect to weight and biases
	# gctw and gctb are numpy arrays
	def backpropagation(self, output):
		gctw = [np.zeros((self.layers[i].weights.shape)) for i in range(1, self.len)]
		gctb = [np.zeros((self.layers[i].biases.shape)) for i in range(1, self.len)]
		#for i in gctw:
		#print("OI {}".format(i))

		# Backward pass, delta from last layer
		# Calculates partial derivative of the cost with respect to the input from layer L
		delta = mse_derivative(self.layers[-1].outputs, output) * sigmoid_derivative(self.layers[-1].z)
		#ERRO AQUIIIIIIIIIIII< DIMENSAO ERRADA
		gctb[-1] = delta
		gctw[-1] = np.dot(delta, self.layers[-2].outputs)		

		# Putting the back in the backpropagation
		for l in reversed(range(1, self.len - 1)):
			# Calculates partial derivative of the cost with respect to the input from layer l
			delta = np.dot(np.transpose(self.layers[l+1].weights)*delta, sigmoid_derivative(self.layers[l].z))
			gctb[l-1] = delta
			gctw[l-1] = np.dot(delta, np.transpose(self.layers[l].outputs))			
			#print("[{}]".format(gctw[l-1]))

		return (gctw, gctb)

	def gradientdescent(self, gctw, gctb, lr):
		for l in range(1, self.len):
			self.layers[l].weights = self.layers[l].weights - lr*gctw[l-1]
			self.layers[l].biases = self.layers[l].biases - lr*gctb[l-1]

	# Trains the NN
	# Computes the backpropagation every single input (SLOW)
	# Using stochastic gradient descent (Every input we update)
	def train(self, training_input, training_output, lr=0.1, epochs=10, iterations=100):
		for i in range(epochs):
			cost = 0
			for _input, _output in zip(training_input, training_output):
				self.feedforward(_input)
				cost = mse(self.layers[-1].outputs, _output)
				#print(cost)
				#gctw, gctb = self.backpropagation(_output)
				#self.gradientdescent(gctw, gctb, lr)
				
			#print("Cost for epoch[{}] = {}".format(i, cost))

	# Returns an np.array with the predictions given a test_set
	def predict(self, test_set):
		prediction = np.zeros(len(test_set))

		for i in range(len(test_set)):
			self.feedforward(test_set[i])
			prediction[i] = self.layers[-1].outputs

		return prediction