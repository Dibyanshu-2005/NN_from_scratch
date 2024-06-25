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
target = np.array([0.3, 0.5])
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


def back_propagation(dc_dzL, eta):
    l = len(network1.layers) - 1  # l = 2
    dc_dz0 = dc_dzL
    prev_output_lines = 0  # Number of lines printed previously

    while loss(network1, target) >= 0.0001:
        while l >= 1:
            dc_dw = np.outer(dc_dz0, al_provider(network1, l - 1))
            dc_db = dc_dz0  # This line was changed
            prev_dc_da = np.dot(dc_dz0, network1.weights[l - 1])

            # Updating weights and biases
            network1.weights[l - 1] -= eta * dc_dw
            network1.bias[l - 1] -= eta * dc_db

            # Print the loss and weights in the same place
            current_loss = loss(network1, target)
            weights_str = "\n".join(
                [f"Layer {i + 1} weights:\n{network1.weights[i]}" for i in range(len(network1.weights))])
            output_str = f"Loss: {current_loss:.6f}\n{weights_str}"

            # Clear the previous output
            sys.stdout.write("\r" + "\033[K")  # Clear the current line
            sys.stdout.write("\033[F" * prev_output_lines)  # Move cursor up to clear previous output
            sys.stdout.write("\033[J")  # Clear all lines below

            # Print the new output
            sys.stdout.write(output_str)
            sys.stdout.flush()

            # Update the number of lines printed
            prev_output_lines = output_str.count('\n') + 1

            dc_dz0 = gradient(prev_dc_da, l - 1)
            l -= 1
        l = len(network1.layers) - 1
        dc_dz0 = dc_dzL

    # Move to the next line after the loop is complete
    print("\nTraining complete")

    return forward(network1)

print(f"Original output: {aL}")
print(f"Target Value: {target}")
print(f"\nOutput after error reduction: {back_propagation(dc_dz, 0.01)}")
print(network1.weights[0])

