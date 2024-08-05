
import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dout):   # dout is the gradient that is floated
        sigmoid_derv = self.output * (1 - self.output)
        return dout * sigmoid_derv

