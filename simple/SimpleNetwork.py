# Based on http://iamtrask.github.io/2015/07/12/basic-python-network/
# Backpropagation in 11 lines

import numpy as np

# sigmoid function
# deriv enables calculating derivate of sigmoid
def nonlin(x, deriv = False):
    if(deriv):
        return x * (1-x)
    return 1/(1+np.exp(-x))

# input training data
inputData = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

# output training data
outputData = np.array([[0, 0, 1, 1]]).T   # Transpose function

# same random seed every time
np.random.seed(1)

# weight matrix
# dimension 3,1
synapse0 = 2 * np.random.random((3, 1)) - 1


for iteration in range(10000):

    # forward propagation
    layer0 = inputData
    layer1 = nonlin(np.dot(layer0, synapse0))

    # errors
    layer1_error = outputData - layer1

    layer1_delta = layer1_error * nonlin(layer1, True)

    synapse0 += np.dot(layer0.T, layer1_delta)

print("after training: ")
print(layer1)





