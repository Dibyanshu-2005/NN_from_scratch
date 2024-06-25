import numpy as np

def fill_matrix(a, b):
    return np.random.rand(a, b)

class Network:
    def __init__(self, num_layers, shape_list):
        self.num_layers = num_layers
        self.layers = shape_list
        self.weights = [fill_matrix(shape_list[i+1], shape_list[i]) for i in range(num_layers-1)]
        self.bias = [np.random.rand(shape_list[i+1]) for i in range(num_layers-1)]  # Corrected bias initialization

# Network parameters
num_layers = int(input("How many layers in NN? : "))
shape_list = [int(input(f"How many neurons in layer {i}: ")) for i in range(num_layers)]

# Create network instance
network1 = Network(num_layers, shape_list)

# Input and target
inp = np.array([float(input(f"Enter the {i} element of input: ")) for i in range(network1.layers[0])])
target = np.array([float(input(f"Enter the {i} element of Target: ")) for i in range(network1.layers[-1])])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def differential_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward(network):
    inp1 = inp
    for i in range(len(network.layers) - 1):
        mat = network.weights[i]
        simple_output = np.dot(inp1, mat.T) + network.bias[i]
        activated_output = sigmoid(simple_output)
        inp1 = activated_output
    return inp1

def zl_provider(network, l1):
    inp1 = inp
    for i in range(l1):
        mat = network.weights[i]
        simple_output = np.dot(inp1, mat.T) + network.bias[i]
        inp1 = simple_output
    return inp1

def al_provider(network, l1):
    inp1 = inp
    for i in range(l1):
        mat = network.weights[i]
        simple_output = np.dot(inp1, mat.T) + network.bias[i]
        activated_output = sigmoid(simple_output)
        inp1 = activated_output
    return inp1

def loss(network0, target1):
    aL0 = forward(network0)
    c = np.sum((target1 - aL0) ** 2) / (2 * len(aL0))
    return c

def gradient(dc_da, layer):
    zl = zl_provider(network1, layer)
    dc_dz = dc_da * differential_sigmoid(zl)
    return dc_dz

aL = forward(network1)
dc_dz = gradient(aL - target, len(network1.layers) - 1)

def back_propagation(dc_dzL, eta):
    l = len(network1.layers) - 1
    dc_dz0 = dc_dzL
    while loss(network1, target) >= 0.0001:
        while l >= 1:
            dc_dw = np.outer(dc_dz0, al_provider(network1, l - 1))
            dc_db = dc_dz0
            prev_dc_da = np.dot(dc_dz0, network1.weights[l - 1])

            # Updating weights and biases
            network1.weights[l - 1] -= eta * dc_dw
            network1.bias[l - 1] -= eta * dc_db

            dc_dz0 = gradient(prev_dc_da, l - 1)
            l -= 1
        l = len(network1.layers) - 1
        dc_dz0 = dc_dzL
    return forward(network1)

print(f"Original output: {aL}")
print(f"Target Value: {target}")
print(f"Output after error reduction: {back_propagation(dc_dz, 0.01)}")
