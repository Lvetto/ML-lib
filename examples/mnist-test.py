from keras.datasets import mnist
from matplotlib import pyplot as plt
from lib import *
from keras.utils import to_categorical
import numpy as np

# Load MNIST dataset from keras and do some pre-processing
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_images = train_images.reshape((train_images.shape[0], 28 * 28))
test_images = test_images.reshape((test_images.shape[0], 28 * 28))

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

print("\n" * 10)

net = supervised_learning((784, 200, 10), [tanh_activation], mse_loss)

losses = []

net.load_weights("weights/mnist.dat")

#losses = net.train(train_images, train_labels, 0.01, 400, 1, 5, "weights/mnist.dat")

#net.save_weights("weights/mnist.dat")

correct = 0
for x,y in zip(test_images, test_labels):
    out = net.compute(x)
    if (np.argmax(out) == np.argmax(y)):
        correct += 1

print(f"Accuracy: {(correct/len(test_images)) * 100:.2f}%")


n = 10
indices = np.random.choice(test_images.shape[0], n * 4, replace=False)
random_images = test_images[indices]
random_labels = test_labels[indices]
network_labels = [net.compute(i) for i in random_images]

for i in range(n * 2):
    plt.subplot(4, n, (i+1) + n * (i // n))
    plt.imshow(random_images[i].reshape((28, 28)), cmap="gray")
    txt = f"Prediction: {np.argmax(network_labels[i])}"
    plt.title(txt)
    plt.axis(False)

    plt.subplot(4, n, (i+1) + n * (i // n + 1))
    plt.title("Confidence levels")
    plt.bar(range(10), network_labels[i])
    plt.xticks(range(10))

plt.show()
