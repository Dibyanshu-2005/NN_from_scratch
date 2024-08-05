import numpy as np

class Sequential:
    def __init__(self, architecture, loss):
        self.architecture = architecture
        self.loss = loss

    def forward(self, x):
        for layer in self.architecture:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.architecture):
            dout = layer.backward(dout)
        return dout

    def fit(self, X, y, n):
        for i in range(n):
            y_pred = self.forward(X)
            loss = self.loss.forward(y_pred, y)
            print(f"Iteration {i+1}/{n}, Loss: {loss}")
            dout = self.loss.backward(y_pred, y)
            self.backward(dout)

    def predict(self, X):
        return self.forward(X)

