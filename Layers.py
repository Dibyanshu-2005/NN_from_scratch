import numpy as np
class Dense:
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.weights = np.random.randn(input_features, output_features) * 0.01
        self.bias = np.zeros((1, output_features))
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        x_flat = x.reshape(-1, self.input_features)  # Flatten to a 2D vector
        self.output = np.dot(x_flat, self.weights) + self.bias
        return self.output

    def backward(self, dout):
        x_flat = self.input.reshape(-1, self.input_features)
        dc_dW = np.dot(x_flat.T, dout)
        dc_db = np.sum(dout, axis=0, keepdims=True)
        dX = np.dot(dout, self.weights.T)
        self.weights -= dc_dW * 0.01  # Adding a small learning rate for stability
        self.bias -= dc_db * 0.01
        return dX.reshape(self.input.shape)  # Reshape back to input shape
