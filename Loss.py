import numpy as np

class QuadLoss:
    def forward(self, y_predicted, y_actual):
        return 0.5 * np.mean((y_predicted - y_actual) ** 2)

    def backward(self, y_predicted, y_actual):
        return y_predicted - y_actual

        # Used in last layer which we can use to find out the next layer's attributes