import numpy as np

class Conv_2d:
    def __init__(self, input_shape, kernel_shape, num_kernels):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.num_kernels = num_kernels
        self.kernels = np.random.randn(num_kernels, kernel_shape[0], kernel_shape[1], input_shape[2])
        self.bias = np.zeros((num_kernels, 1))
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        h_in, w_in, d_in = x.shape  # using .shape to directly get the shape
        h_k, w_k, d_k = self.kernel_shape
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
        h_out, w_out, _ = dout.shape
        h_in, w_in, d_in = self.input.shape
        h_k, w_k, d_k = self.kernel_shape

        dX = np.zeros_like(self.input)
        dK = np.zeros_like(self.kernels)
        db = np.zeros_like(self.bias)

        for k in range(self.num_kernels):
            for i in range(h_out):
                for j in range(w_out):
                    region = self.input[i:i + h_k, j:j + w_k, :]
                    dK[k] += region * dout[i, j, k]
                    db[k] += dout[i, j, k]
                    dX[i:i + h_k, j:j + w_k, :] += self.kernels[k] * dout[i, j, k]

        self.kernels -= dK  # Assuming a learning rate of 1 for simplicity
        self.bias -= db
        return dX


