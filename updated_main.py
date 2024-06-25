import numpy as np

num_layers = int(input("How many layers in NN? : "))
shape_list = [int(input(f"How many neurons in layer {i}: ")) for i in range(num_layers)]
# for example shape_list = [4,3,2]


def fill_matrix(a, b):    # Using this we can fill matrix of size (a,b)
    return np.random.rand(a, b)

class Network:
    def __init__(self):
        self.layers = num_layers
        self.shape = shape_list
        self.weights = [fill_matrix(shape_list[i+1], shape_list[i]) for i in range(num_layers-1)]
        self.bias = [fill_matrix(1, shape_list[i+1]) for i in range(num_layers-1)]


network1 = Network()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(network, inp):
    inp1 = inp
    for i in range(0, network.layers-1):
        mat = network.weights[i]
        transposed_mat = mat.transpose()
        simple_output = np.dot(inp1, transposed_mat) + network.bias[i]
        activated_output = sigmoid(simple_output)
        inp1 = activated_output
    print(f"Output : {inp1}")


first_input = [float(input(f"Enter the {i} element of first input: ")) for i in range(shape_list[0])]
forward(network1, np.array(first_input))
