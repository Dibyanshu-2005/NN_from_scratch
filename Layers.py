import numpy as np

class Dense:
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.weights = np.random.randn(input_features, output_features)  # Weight matrix prepared
        self.bias = np.random.rand(1, output_features)  # bias matrix prepared
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward(self, dout):
        dc_dW = np.dot(self.input.T, dout)
        dc_db = np.sum(dout, axis=0, keepdims=True)
        dX = np.dot(dout, self.weights.T)
        self.weights -= dc_dW
        self.bias -= dc_db
        return dX



