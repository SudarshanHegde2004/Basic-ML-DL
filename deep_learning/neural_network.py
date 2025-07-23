import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.loss_prime = None

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                err += self.loss(y_train[j], output)

                # Backward propagation
                gradient = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            err /= samples
            if i % 100 == 0:
                print(f'epoch {i + 1}/{epochs}   error={err}')
