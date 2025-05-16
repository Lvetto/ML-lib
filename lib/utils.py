import numpy as np
from matplotlib import pyplot as plt

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
    
 
