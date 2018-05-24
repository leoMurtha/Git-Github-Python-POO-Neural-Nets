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
		#print(0, self.layers[0].outputs.shape)
		for k in range(1, self.len):
			self.layers[k].z = np.dot(self.layers[k].weights, self.layers[k - 1].outputs)
			
			#print("<{}> * <{}> + <{}> = {}\n\n".format(np.shape(self.layers[k].weights), np.shape(self.layers[k - 1].outputs) , np.shape(np.transpose(self.layers[k].biases)), np.shape(self.layers[k].z)))
			self.layers[k].z = np.add(self.layers[k].z, np.transpose(self.layers[k].biases))
			#print("<{}> * <{}> + <{}> = {}\n\n".format(np.shape(self.layers[k].weights), np.shape(self.layers[k - 1].outputs) , np.shape(np.transpose(self.layers[k].biases)), np.shape(self.layers[k].z)))
	
			self.layers[k].outputs = sigmoid(self.layers[k].z)
			#print((self.layers[k].z), self.layers[k].outputs)
			#print(k, self.layers[k].outputs.shape, self.layers[k].z.shape)
			
	# Returns a tuple (gctw, gctb) wich are
	# The gradients of cost function with respect to weight and biases
	# gctw and gctb are numpy arrays
	def backpropagation(self, output):
		gctw = [np.zeros((self.layers[i].weights.shape)) for i in range(1, self.len)]
		gctb = [np.zeros((self.layers[i].biases.shape)) for i in range(1, self.len)]
		
		# Backward pass, delta from last layer
		# Calculates partial derivative of the cost with respect to the input from layer L
		delta = np.multiply(mse_derivative(self.layers[-1].outputs, output), sigmoid_derivative(self.layers[-1].z)) 
		gctb[-1] = delta
		gctw[-1] = np.transpose(np.dot(delta, np.transpose(self.layers[-2].outputs)))		
		
		# Putting the back in the backpropagation
		for l in reversed(range(1, self.len - 1)):
			#FALTA TERMINAR E VER SHAPES DO PESO E ETC TA INDO AIII
			delta = np.multiply(np.dot(np.transpose(self.layers[l+1].weights), delta), sigmoid_derivative(self.layers[l].z)) 
			#print(delta.shape)	
			gctb[l-1] = delta
			#print(self.layers[l-1].outputs.shape, delta.shape)
			gctw[l-1] = np.dot(self.layers[l-1].outputs, np.transpose(delta))			
			#print(gctw[l-1].shape)
		
		return (gctw, gctb)

	def gradientdescent(self, gctw, gctb, lr):
		for k in range(1, self.len):
			#print("Weightss:{}".format(self.layers[l].weights))
			#print("Gweights:{}".format(np.transpose(gctw[l-1])))
			#print(self.layers[l].weights.shape, gctw[l-1].shape)
			#print(self.layers[l].biases.shape, gctb[l-1].shape)
			#print("A-{} WEIGHT[{}] GRADIENT[{}]".format(k, self.layers[k].weights.shape, np.transpose(lr*gctw[k-1]).shape))
			#print("A-{} BIASE[{}] GRADIENT[{}]".format(k, self.layers[k].biases.shape, np.transpose(lr*gctb[k-1]).shape))
			self.layers[k].weights = np.subtract(self.layers[k].weights, np.transpose(lr*gctw[k-1]))
			self.layers[k].biases = np.subtract(self.layers[k].biases, np.transpose(lr*gctb[k-1]))
			#print("B-{} WEIGHT[{}] GRADIENT[{}]".format(k, self.layers[k].weights.shape, np.transpose(lr*gctw[k-1]).shape))
			#print("B-{} BIASE[{}] GRADIENT[{}]".format(k, self.layers[k].biases.shape,  np.transpose(lr*gctb[k-1]).shape))

	# Trains the NN
	# Computes the backpropagation every single input (SLOW)
	# Using stochastic gradient descent (Every input we update)
	def train(self, training_input, training_output, lr=0.1, epochs=10):
		
		for i in range(epochs):
			cost = 0
			for _input, _output in zip(training_input, training_output):
				_input = np.reshape(_input, (len(_input), 1))
				_output = np.reshape(_output, (len(_output), 1))
				
				#print("INPUT[{}] OUTPUT[{}]".format(_input.shape, _output.shape))
				self.feedforward(_input)
				#print(mse(self.layers[-1].outputs, _output))
				cost += mse(self.layers[-1].outputs, _output)/len(training_input)
				gctw, gctb = self.backpropagation(_output)
				self.gradientdescent(gctw, gctb, lr)
				
			print("Cost for epoch {} = {:2.5f}".format(i+1, cost[0, 0]))
			
	# Returns an np.array with the predictions given a test_set
	def predict(self, X_test, y_test):
		pred = np.zeros(y_test.shape)
		for i in range(len(X_test)):
			self.feedforward(np.reshape(X_test[i], (len(X_test[i]), 1)))
			pred[i] = (self.layers[-1].outputs)

		return pred