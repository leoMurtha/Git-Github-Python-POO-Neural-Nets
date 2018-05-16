import numpy as np
from neural_network import *

def main():

	# XOR data set
	inp = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
	#print(np.shape(inp))

	outp = np.array([1, 0, 0, 1])

	nn = NeuralNetwork()

	nn.add(Layer(2, 2))
	
	nn.add(Layer(2, 2))
	
	nn.add(Layer(1, 2))

	nn.train(inp, outp, 0.01, 200000)
	
	test_set = np.array([[0, 1], [1, 1], [0, 1], [1, 0]])
	nn.predict(test_set)

if __name__ == '__main__':
	main()