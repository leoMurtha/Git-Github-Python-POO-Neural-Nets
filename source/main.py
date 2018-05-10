import numpy as np
from neural_network import *

def main():

	# XOR data set
	inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	outp = np.array([0, 1, 1, 0])
	
	nn = NeuralNetwork()

	nn.add(Layer(2, 2))
	nn.add(Layer(2, 2))

	nn.train(inp, outp, 0.01, 100, 1)

if __name__ == '__main__':
	main()