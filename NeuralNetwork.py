import random
import numpy as np
#http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
class NeuralNetwork(object):
    def __init__(self, sizes):
        self.numOfLayers = len(sizes)
        self.sizes = sizes
        # standard normal distribution
        # multidimensional NUMPY ARRAY where first array is size 3 and last array is size 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # [2,3,1] -> [(2,3), (3,1)] -> 1st, 2nd  2nd, 3rd
        # [[a,a], [a,a], [a,a]] [[a,a,a]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

#Numpy automatically applies the function sigmoid elementwise
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

#connections between layers are only allowed in forward direction
def feedforward(self, input):
    """Return the output of the network if "a" is input."""
    for bias, weight in zip(self.biases, self.weights):
        input = sigmoid(np.dot(weight, input) + bias)
    return input

# trainingData = (x,y) where x = training inputs, y = desired output
def gradientDescent(self, trainingData, epochs, miniBatchSize, learningRate, testData = None):
    for i in range(epochs):
        random.shuffle(trainingData)
        miniBatches = [trainingData[k: k + miniBatchSize] for k in range(0, len(trainingData), miniBatchSize)]
        for miniBatch in miniBatches:
            # single step of gradient descent
            self.updateMiniBatch(miniBatch, learningRate)
        if testData:
            print("Epoch {} : {} / {}".format(i, self.evaluate(testData), len(testData)));
        else:
            print("Epoch {} complete".format(i))


def updateMiniBatch(self, miniBatch, learningRate):
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    is the learning rate."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in miniBatch:
        # https://pythonmachinelearning.pro/complete-guide-to-deep-neural-networks-part-2/
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w - (learningRate / len(miniBatch)) * nw
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (learningRate / len(miniBatch)) * nb
                   for b, nb in zip(self.biases, nabla_b)]

def backprop(self, x, y):
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))