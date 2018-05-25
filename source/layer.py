import numpy as np


class Layer:
    def __init__(self, num_of_neurons=None, neuron_dim=None, outputs=None):
        """
        Layer constructor
        It initializes a layer with random weights and biases
        Using He-et-al Initialization
        neuron_dim is the i.e number of neurons in the previous layer
        if neurom_dim is null then it is a input layer
        """
        self.num_of_neurons = num_of_neurons
        self.neuron_dim = neuron_dim
        if(neuron_dim != None):
            self.weights = np.array(np.random.random((num_of_neurons, neuron_dim)) * np.sqrt(2. / neuron_dim))
            self.biases = np.array(np.random.random((1, num_of_neurons)) / 2 * np.random.randn(1 ,num_of_neurons))
        else:
            self.weights = self.biases = 0
        self.outputs = outputs
        self.delta = 0

    def log(self):
        """Prints out layer settings"""
        if(self.neuron_dim > 0):
            print("Number of neurons [{}]\nNumber of inputs per neuron [{}]".format(self.num_of_neurons, self.neuron_dim))
            print("Weights {}:\n {}\n".format(np.shape(self.weights), self.weights))
            print("Biases {}:\n {}\n".format(np.shape(self.biases), self.biases))
        else:
            print("<<Input Layer>> Number of features [{}]".format(self.neuron_dim))
            for output in self.outputs:
                print(output)
