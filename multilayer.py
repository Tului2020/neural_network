import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import Layer

nnfs.init()

X = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8],
]

X, y = spiral_data(100, 3)

layer1 = Layer.Layer_Dense(2, 5)
activation1 = Layer.Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
print(activation1.output)

