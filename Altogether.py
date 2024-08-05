import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dout):
        sigmoid_derv = self.output * (1 - self.output)
        return dout * sigmoid_derv


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


class Conv_2d:
    def __init__(self, input_shape, kernel_shape, num_kernels):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.num_kernels = num_kernels
        self.kernels = np.random.randn(num_kernels, *kernel_shape, input_shape[2]) * 0.01
        self.bias = np.zeros((num_kernels, 1))
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        h_in, w_in, _ = x.shape
        h_k, w_k = self.kernel_shape
        h_out = h_in - h_k + 1
        w_out = w_in - w_k + 1
        self.output = np.zeros((h_out, w_out, self.num_kernels))

        for k in range(self.num_kernels):
            for i in range(h_out):
                for j in range(w_out):
                    region = x[i:i + h_k, j:j + w_k, :]
                    self.output[i, j, k] = np.sum(region * self.kernels[k]) + self.bias[k]

        return self.output

    def backward(self, dout):
        h_out, w_out, num_kernels = dout.shape
        h_in, w_in, d_in = self.input_shape
        h_k, w_k = self.kernel_shape

        dX = np.zeros_like(self.input)
        dK = np.zeros_like(self.kernels)
        db = np.zeros_like(self.bias)

        for k in range(num_kernels):
            for i in range(h_out):
                for j in range(w_out):
                    region = self.input[i:i + h_k, j:j + w_k, :]
                    dK[k] += region * dout[i, j, k]
                    db[k] += dout[i, j, k]
                    dX[i:i + h_k, j:j + w_k, :] += self.kernels[k] * dout[i, j, k]

        self.kernels -= dK * 0.01  # Adding a small learning rate for stability
        self.bias -= db * 0.01
        return dX


class QuadLoss:
    def forward(self, y_predicted, y_actual):
        return 0.5 * np.mean((y_predicted - y_actual) ** 2)

    def backward(self, y_predicted, y_actual):
        return y_predicted - y_actual


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


# Example usage
architecture = [
    Conv_2d(input_shape=(28, 28, 1), kernel_shape=(3, 3), num_kernels=8),
    Sigmoid(),
    Dense(input_features=26*26*8, output_features=10)
]

model = Sequential(architecture, QuadLoss())

X = np.random.randn(28, 28, 1)  # Single sample of 28x28 grayscale image
y = np.random.randn(10)  # Single sample of 10 output classes

model.fit(X, y, n=10)

prediction = model.predict(X)
print("Prediction:", prediction)
