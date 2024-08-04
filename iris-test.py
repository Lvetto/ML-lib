import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lib import *
from matplotlib import pyplot as plt
from numpy import tanh

# load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot label encoding
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

net = supervised_learning((4, 4, 3), tanh_activation, mse_loss)

losses = net.train(X_train, y_train, 0.1, 1000)

def evaluate(network, X_test, y_test):
    correct = 0
    total = len(y_test)
    for input, target in zip(X_test, y_test):
        output = network.compute(input)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

evaluate(net, X_test, y_test)

plt.plot(losses)
plt.show()

