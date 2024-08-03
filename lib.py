import numpy as np
from matplotlib import pyplot as plt
from time import time
import pickle

class network:
    def __init__(self, layer_sizes, activation=np.tanh, activation_derivative=None):
        # Assign an activation function and its derivative. As a default tanh is used
        self.activation = activation
        self.activation_derivative = (
            activation_derivative if activation_derivative is not None
            else lambda x: 1 - np.tanh(x)**2
        )
        
        # Initialize layer biases and weights
        self.biases = [np.random.rand(size) for size in layer_sizes[1:]]
        self.weights = [
            np.random.rand(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ]

    def compute(self, input):
        # Ensure input is a numpy array
        input = np.array(input)
        
        # Store values in the network layer by layer, starting from input
        self.layers = [input]
        
        # Forward pass through each layer
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = self.layers[-1] @ weight + bias
            activation = self.activation(z)
            self.layers.append(activation)
        
        return self.layers[-1]
    
    def loss(self, output, target):
        return 0.5 * np.sum((output - target)**2)   # always positive and has a nice derivative
    
    def backpropagate(self, input, target, learning_rate):
        # Forward pass
        output = self.compute(input)
        
        # Calculate error for the output layer
        output_error = (output - target) * self.activation_derivative(output)
        
        # Initialize error list for each layer
        errors = [None] * len(self.layers)
        errors[-1] = output_error
        
        # Compute derivatives and apply weight/biases corrections
        for l in reversed(range(len(self.weights))):
            # Update weights and biases
            errors[l] = (errors[l + 1] @ self.weights[l].T) * self.activation_derivative(self.layers[l])
            self.weights[l] -= learning_rate * np.outer(self.layers[l], errors[l + 1])
            self.biases[l] -= learning_rate * errors[l + 1]

    def train(self, inputs, targets, learning_rate, epochs, update_interval=10, save_interval=None):
        losses = []
        start_time = time()
        for epoch in range(epochs):
            total_loss = 0
            for input, target in zip(inputs, targets):
                self.backpropagate(input, target, learning_rate)
                total_loss += self.loss(self.layers[-1], target)
            losses.append(total_loss)
            if (not epoch % update_interval):
                progress = (epoch + 1) / epochs
                time_taken = time() - start_time
                avg_loss = total_loss / len(inputs)
                eta = ((time_taken / (epoch +1)) * epochs) - time_taken
                print(f"Epoch: {epoch+1}, Progress: {progress * 100:.2f}%, Average loss: {avg_loss:.4f}, Time taken: {time_taken:.2f}s, Eta: {eta:.2f}s\r", end="")
            
            if (save_interval is not None):
                if (not epoch % save_interval):
                    self.save_weights("weights.dat")
        print("\n")
        return losses

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



def draw_network(net : network):
    layers = net.layers
    edges = net.edge_matrices

    # each node is represented by a dot and has a text annotation showing its value
    for n, layer in enumerate(layers):
        xs = [n] * len(layer)
        ys = [i for i,_ in enumerate(layer)]
        plt.plot(xs, ys, "b.")

        for i,t in enumerate(zip(xs, ys)):
            x, y = t
            plt.text(x, y + 0.05, layer[i])

    m = np.max([np.max(np.abs(i)) for i in edges])
    
    # edges are represented by lines. A darker line represents a lower value (in abs, compared to all others)
    for n, mat in enumerate(edges):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] != 0:
                    t = abs(mat[i, j] / (m * 1.2))
                    rgb_col = (t, t, t)
                    plt.plot([n, n+1], [i, j], "-", color=rgb_col)
    
 
