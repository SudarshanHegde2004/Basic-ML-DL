import numpy as np

class Conv2D:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        self.output = np.zeros((input.shape[0], 
                              input.shape[1] // self.pool_size,
                              input.shape[2] // self.pool_size))
        
        for i in range(0, input.shape[1], self.pool_size):
            for j in range(0, input.shape[2], self.pool_size):
                self.output[:, i//self.pool_size, j//self.pool_size] = \
                    np.max(input[:, i:i+self.pool_size, j:j+self.pool_size], axis=(1,2))
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        
        for i in range(0, self.input.shape[1], self.pool_size):
            for j in range(0, self.input.shape[2], self.pool_size):
                patch = self.input[:, i:i+self.pool_size, j:j+self.pool_size]
                mask = patch == np.max(patch, axis=(1,2))[:, np.newaxis, np.newaxis]
                input_gradient[:, i:i+self.pool_size, j:j+self.pool_size] = \
                    mask * output_gradient[:, i//self.pool_size, j//self.pool_size][:, np.newaxis, np.newaxis]
        return input_gradient
