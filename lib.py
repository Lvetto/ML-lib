import numpy as np
from matplotlib import pyplot as plt
from time import time
import pickle
from random import random

class base_network:
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

class supervised_learning(base_network):
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

class reinforcement_learning(supervised_learning):
    def __init__(self, layer_sizes, activations, loss):
        super().__init__(layer_sizes, activations, loss)

    # make the wrong choice on purpose during training with probability epsilon to promote exploration
    def choose_action(self, q, epsilon):
        if random() < epsilon:
            return np.random.choice(len(q))
        else:
            return np.argmax(q)
    
    def train(self, initial_state, transition_function, reward, learning_rate, discount, epsilon, epochs, decay_rate=0.9):
        # the agent can move between states by making choices (represented as a confidence level vector that should approach the expected value for each in a given state)
        # to train it, we give it an initial state and have it make a choice and simulate the outcome
        current_state = initial_state
        for epoch in range(epochs):
            # make a choice based on the current state and current weights
            q = self.compute(current_state)
            choice = self.choose_action(q, epsilon) # sometimes we ignore it

            # simulate the outcome and what the agent would do in the next state with the current weights
            next_state = transition_function(current_state, choice)
            q_next = self.compute(next_state)

            # approximate target output using the Belmann equation
            target = q.copy()
            target[choice] = reward(current_state, choice) + discount * np.max(q_next)  # we only consider the component corresponding to the choice made to increase stability

            # adjust the weights using backpropagation and advace to the next state
            self.backpropagate(current_state, target, learning_rate)
            current_state = next_state

            # decrease randomness over time. More randomness helps with training time in the beginning, but is not needed later
            epsilon *= decay_rate


# used to pass a function toghether with its derivative when instantiating a network. Saves on some parameters in the constructors
class FunctionWithDerivative:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, *args):
        return self.func(*args)

# some common activation functions with their derivatives (mostly from chatgpt)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

sigmoid_activation = FunctionWithDerivative(sigmoid, sigmoid_derivative)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

relu_activation = FunctionWithDerivative(relu, relu_derivative)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

tanh_activation = FunctionWithDerivative(tanh, tanh_derivative)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

leaky_relu_activation = FunctionWithDerivative(
    lambda x: leaky_relu(x, alpha=0.01),
    lambda x: leaky_relu_derivative(x, alpha=0.01)
)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilize with max trick
    return exp_x / np.sum(exp_x, axis=0)

def softmax_derivative(x):
    # Assumes x is the softmax output
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

softmax_activation = FunctionWithDerivative(softmax, softmax_derivative)

# some common loss functions with derivatives (mostly from chatgpt)

mse_loss = FunctionWithDerivative(lambda output, target: 0.5 * np.sum((output - target)**2), lambda output, target: output - target)

# Cross-Entropy
def cross_entropy_loss(output, target):
    epsilon = 1e-15     # used to avoid log(0) singularity
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.sum(target * np.log(output)) / output.shape[0]

def cross_entropy_loss_derivative(output, target):
    # Assumes output is softmax probabilities
    return (output - target) / output.shape[0]

cross_entropy_loss_function = FunctionWithDerivative(cross_entropy_loss, cross_entropy_loss_derivative)

#Mean Absolute Error (MAE)
def mae_loss(output, target):
    return np.mean(np.abs(output - target))

def mae_loss_derivative(output, target):
    return np.where(output > target, 1, -1) / output.size

mae_loss_function = FunctionWithDerivative(mae_loss, mae_loss_derivative)

# Hinge Loss
def hinge_loss(output, target):
    return np.mean(np.maximum(0, 1 - target * output))

def hinge_loss_derivative(output, target):
    return np.where(target * output < 1, -target, 0) / output.size

hinge_loss_function = FunctionWithDerivative(hinge_loss, hinge_loss_derivative)


# draw a network using a pyplot scatterplot. Not particularly useful anymore...
def draw_network(net):
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
    
 
