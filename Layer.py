import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons) / 10
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
