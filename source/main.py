import numpy as np
from neural_network import *

def main():

	# XOR data set
	inp =[[1, 0, 1]]
	
	outp = [0]

	
	nn = NeuralNetwork()

	nn.add(Layer(2, 3))
	nn.add(Layer(1, 2))

	nn.train(inp, outp, 0.01, 10, 1)
	

	#test_set = np.array([[0, 1], [1, 1], [1, 1], [1, 0]])
	#prediction = nn.predict(test_set)
	
	#print(prediction)

if __name__ == '__main__':
	main()