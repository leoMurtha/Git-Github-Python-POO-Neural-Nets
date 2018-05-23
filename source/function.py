import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return 1 - (tanh(x)**2)


def mse(y, yT):
    return np.subtract(y,yT)**2


def mse_derivative(y, yT):
	return np.subtract(y, yT)*2.0


"""
numGrad = np.zeros(np.shape(params))
				pertub = np.zeros(np.shape(params))

				e = 1e-4
				for p in range(len(params)):
					pertub[p] = e
					self.layers[1].weights = params + e
					self.feedforward(_input)
					co1 = mse(self.layers[-1].outputs, _output)/len(training_input)

					self.layers[1].weights = params - e
					self.feedforward(_input)
					co2 = mse(self.layers[-1].outputs, _output)/len(training_input)
					
					#print(co2, co1)

					numGrad[p] = (co2-co1)/2*e

					pertub[p] = 0

				self.layers[1].weights = params

				print(np.linalg.norm(gctw[0] - numGrad)/np.linalg.norm(gctw[0] + numGrad))
				
"""
