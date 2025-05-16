import numpy as np
import pickle

class neural_network:
    def __init__(self, layer_sizes, activations):
        # Assign an activation functions and their derivatives
        if (len(activations) != len(layer_sizes) -1):
            self.activations = [activations[0]] * (len(layer_sizes) -1)
            print("Wrong number of activation functions. Using the first one for each layer")
        else:
            self.activations = activations
        
        # Initialize layer biases and weights
        self.biases = [np.random.rand(size) for size in layer_sizes[1:]]
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def compute(self, input):
        # Ensure input is a numpy array
        input = np.array(input)
        
        # Store values in the network layer by layer, starting from input
        self.layers = [input]
        
        # Forward pass through each layer
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = self.layers[-1] @ weight + bias
            z = self.activations[i](z)
            self.layers.append(z)

        return self.layers[-1]

    def save_weights(self, filename):
        #Saves the weights and biases to a file
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)

    def load_weights(self, filename):
        #Loads the weights and biases from a file
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
