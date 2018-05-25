import numpy as np
from layer import *
from function import *
from sklearn.metrics import confusion_matrix
		
class NeuralNetwork(object):
	"""A neural network composed of layers"""

	def __init__(self):
		"""
		Creates a Neural Net with a empty input layer
		"""
		super(NeuralNetwork, self).__init__()
		self.layers = [Layer()]
		self.len = 1

	def add(self, layer):
		"""
		Adds a layer to the NN
		"""
		self.layers.append(layer)
		self.len += 1

	def feedforward(self, input):
		"""
		Passes the input through the entire network
		"""
		self.layers[0] = Layer(num_of_neurons=len(input), outputs=input)
		for k in range(1, self.len):
			self.layers[k].z = np.dot(self.layers[k].weights, self.layers[k - 1].outputs)
			self.layers[k].z = np.add(self.layers[k].z, np.transpose(self.layers[k].biases))
			self.layers[k].outputs = sigmoid(self.layers[k].z)
			
	def backpropagation(self, output):
		"""
		Returns a tuple (gctw, gctb) wich are
		The gradients of cost function with respect to weight and biases
		gctw and gctb are numpy arrays
		"""
		gctw = [np.zeros((self.layers[i].weights.shape)) for i in range(1, self.len)]
		gctb = [np.zeros((self.layers[i].biases.shape)) for i in range(1, self.len)]
		
		# Backward pass, delta from last layer
		# Calculates partial derivative of the cost with respect to the input from layer L
		delta = np.multiply(mse_derivative(self.layers[-1].outputs, output), sigmoid_derivative(self.layers[-1].z)) 
		gctb[-1] = delta
		gctw[-1] = np.transpose(np.dot(delta, np.transpose(self.layers[-2].outputs)))		
		
		# Putting the back in the backpropagation
		# delta = weights(l+1)T . delta(l+1) HADDAMAD g'(z(l))
		# gradient = al-1 . delta(l)T
		for l in reversed(range(1, self.len - 1)):
			delta = np.multiply(np.dot(np.transpose(self.layers[l+1].weights), delta), sigmoid_derivative(self.layers[l].z)) 
			gctb[l-1] = delta
			gctw[l-1] = np.dot(self.layers[l-1].outputs, np.transpose(delta))			
			
		return (gctw, gctb)


	def gradientdescent(self, gctw, gctb, lr):
		"""
		Updates the biases and weights using gradients
		"""
		for k in range(1, self.len):
			self.layers[k].weights = np.subtract(self.layers[k].weights, np.transpose(lr*gctw[k-1]))
			self.layers[k].biases = np.subtract(self.layers[k].biases, np.transpose(lr*gctb[k-1]))
			
	def train(self, training_input, training_output, lr=0.1, epochs=10):
		"""
		Trains the NN
		Computes the backpropagation every single input (SLOW)
		Using stochastic gradient descent (Every input we update)
		"""
		for i in range(epochs):
			cost = 0
			for _input, _output in zip(training_input, training_output):
				# Formatting input and output
				_input = np.reshape(_input, (len(_input), 1))
				_output = np.reshape(_output, (len(_output), 1))

				self.feedforward(_input)
				cost += mse(self.layers[-1].outputs, _output)/len(training_input)
				gctw, gctb = self.backpropagation(_output)
				self.gradientdescent(gctw, gctb, lr)
				
			print("Cost for epoch {} = {:2.5f}".format(i+1, cost[0, 0]))
			
	def predict(self, X_test, y_test):
		"""
		Returns an np.array with the predictions given a test_set 
		"""
		pred = np.zeros(y_test.shape)
		for i in range(len(X_test)):
			self.feedforward(np.reshape(X_test[i], (len(X_test[i]), 1)))
			pred[i] = (self.layers[-1].outputs).reshape(pred[i].shape)

		return pred

	def score(self, y_test, y_pred):
		cm = confusion_matrix(y_test, y_pred)
		print("Net's Accuracy {:2.2f}%".format((cm[0, 0] + cm[1, 1])/np.sum(cm)*100.0))