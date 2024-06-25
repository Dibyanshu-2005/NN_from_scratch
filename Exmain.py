import numpy as np
import time
import sys


class Network:
    def __init__(self):
        self.layers = [4, 3, 2]
        self.weights = [np.array([[0.1, 0.2, 0.3, 0.4],
                                  [0.5, 0.6, 0.7, 0.8],
                                  [0.1, 0.2, 0.3, 0.4]]),
                        np.array([[0.1, 0.2, 0.3],
                                  [0.4, 0.5, 0.6]])]
        self.bias = [np.array([0.12, 0.13, 0.14]), np.array([0.15, 0.16])]


network1 = Network()
inp = np.array([0.1, 0.2, 0.3, 0.4])
target = np.array([0.1, 0.2])
network1.weights = [w * 0.001 for w in network1.weights]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def differential_sigmoid(x):  # differentiation of sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def forward(network):  # forward function
    inp1 = inp
    for i in range(0, len(network.layers) - 1):
        mat = network.weights[i]
        transposed_mat = mat.transpose()
        simple_output = np.dot(inp1, transposed_mat) + network.bias[i]
        activated_output = sigmoid(simple_output)
        inp1 = activated_output
    return inp1


def zl_provider(network, l1):  # provides zl
    inp1 = inp
    for i in range(0, l1):
        mat = network.weights[i]
        transposed_mat = mat.transpose()
        simple_output = np.dot(inp1, transposed_mat) + network.bias[i]
        inp1 = simple_output
    return inp1


def al_provider(network, l1):
    inp1 = inp
    for i in range(0, l1):
        mat = network.weights[i]
        transposed_mat = mat.transpose()
        simple_output = np.dot(inp1, transposed_mat) + network.bias[i]
        activated_output = sigmoid(simple_output)
        inp1 = activated_output
    return inp1


aL = forward(network1)


def loss(network0, target1):  # provides loss for tracking
    c = 0
    aL0 = forward(network0)
    for i in range(len(target1)):
        c += (target1[i] - aL0[i]) ** 2
    return c / (2*len(aL0))


def gradient(dc_da, layer):  # Provides dc/dz
    zl = zl_provider(network1, layer)
    dc_dz = np.array([(dc_da[j]) * (differential_sigmoid(zl[j])) for j in range(len(dc_da))])
    return dc_dz


dc_dz = gradient(aL-target, len(network1.layers)-1)  # for the last layer


import numpy as np
import sys
import time

def back_propagation(dc_dzL, eta):
    l = len(network1.layers) - 1  # l = 2
    dc_dz0 = dc_dzL

    min_loss = float('inf')
    best_weights = None
    best_biases = None

    while True:
        l = len(network1.layers) - 1  # Reset l to the last layer index for each epoch
        dc_dz0 = dc_dzL  # Reset dc_dz0 for each epoch

        while l >= 1:
            dc_dw = np.outer(dc_dz0, al_provider(network1, l - 1))
            dc_db = dc_dz0
            prev_dc_da = np.dot(dc_dz0, network1.weights[l - 1].T)  # Transpose to match dimensions

            # Updating weights and biases
            network1.weights[l - 1] -= eta * dc_dw
            network1.biases[l - 1] -= eta * dc_db  # Use biases[l - 1] instead of bias[l - 1]

            dc_dz0 = gradient(prev_dc_da, l - 1)
            l -= 1

        # Calculate the current loss
        current_loss = loss(network1, target)
        sys.stdout.write(f"\rLoss: {current_loss:.6f}")
        sys.stdout.flush()
        time.sleep(0.001)

        # Check if the current loss is less than the minimum loss
        if current_loss < min_loss:
            min_loss = current_loss
            best_weights = [w.copy() for w in network1.weights]
            best_biases = [b.copy() for b in network1.biases]
        else:
            # Restore the best weights and biases and stop training
            network1.weights = best_weights
            network1.biases = best_biases
            break

    return forward(network1)


print(f"Original output: {aL}")
print(f"Target Value: {target}")
print(f"\nOutput after error reduction: {back_propagation(dc_dz, 0.01)}")

