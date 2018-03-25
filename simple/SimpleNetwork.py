# Based on http://iamtrask.github.io/2015/07/12/basic-python-network/
# neural networks in 11 lines

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
outputData = np.array([[0, 0, 1, 1]]).T   # Transpose function: flips over axle

# same random seed every time
np.random.seed(1)

# weight matrix
# dimension 3,1
synapse0 = 2 * np.random.random((3, 1)) - 1

print("synapse0: \n" + str(synapse0))

for iteration in range(10000):

    # forward propagation: get the output and compare it with the real value to get the error.
    layer0 = inputData
    layer1 = nonlin(np.dot(layer0, synapse0))

    # errors
    layer1_error = outputData - layer1

    # error weighted derivative
    # layer1_error = (4,1) matrix
    # nonlin(layer1, True) returns (4,1) matrix
    '''
    When we multiply the "slopes" by the error, we are reducing the error 
    of high confidence predictions. Look at the sigmoid picture again!
    If the slope was really shallow (close to 0), then the network either
    had a very high value, or a very low value. This means that the network 
    was quite confident one way or the other. However, if the network guessed 
    something close to (x=0, y=0.5) then it isn't very confident.
    '''
    layer1_delta = layer1_error * nonlin(layer1, True)
    synapse0 += np.dot(layer0.T, layer1_delta)

    # layer1 = hidden layer
    if (iteration == 1 or iteration==9999):
        print("layer1: \n" + str(layer1))

print("synapse0: \n" + str(synapse0))





