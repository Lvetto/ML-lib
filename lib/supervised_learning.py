import numpy as np
from time import time
from lib.network import *


class supervised_learning(neural_network):
    def __init__(self, layer_sizes, activations, loss):
        super().__init__(layer_sizes, activations)
        self.loss = loss

    def backpropagate(self, input, target, learning_rate):
        # Forward pass
        output = self.compute(input)
        
        # Calculate error for the output layer
        output_error = self.loss.derivative(output, target) * self.activations[-1].derivative(output)
        
        # Initialize error list for each layer
        errors = [None] * len(self.layers)
        errors[-1] = output_error
        
        # Compute derivatives and apply weight/biases corrections
        for l in reversed(range(len(self.weights))):
            # Update weights and biases
            errors[l] = (errors[l + 1] @ self.weights[l].T) * self.activations[l].derivative(self.layers[l])
            self.weights[l] -= learning_rate * np.outer(self.layers[l], errors[l + 1])
            self.biases[l] -= learning_rate * errors[l + 1]

    def train(self, inputs, targets, learning_rate, epochs, update_interval=10, save_interval=None, save_path=None):
        losses = []
        start_time = time()
        for epoch in range(epochs):
            total_loss = 0
            # show the network all the inputs in the training set, then adjust the weights using the target outputs from the same set
            for input, target in zip(inputs, targets):
                self.backpropagate(input, target, learning_rate)
                total_loss += self.loss(self.layers[-1], target)
            losses.append(total_loss)
  
            # periodically give updates with some useful info
            if (not epoch % update_interval):
                progress = (epoch + 1) / epochs
                time_taken = time() - start_time
                avg_loss = total_loss / len(inputs)
                eta = ((time_taken / (epoch +1)) * epochs) - time_taken
                print(f"Epoch: {epoch+1}, Progress: {progress * 100:.2f}%, Average loss: {avg_loss:.4f}, Time taken: {time_taken:.2f}s, Eta: {eta:.2f}s\r", end="")
        
            # if a save path is given, periodically dump the weights into a file
            if (save_interval is not None and save_path is not None):
                if (not epoch % save_interval):
                    self.save_weights("weights.dat")

        print("\n")
        return losses