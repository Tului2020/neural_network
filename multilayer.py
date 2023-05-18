import numpy as np
import nnfs
import Layer

nnfs.init()

X = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8],
]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# activation = Layer.Activation_ReLU

# print(activation.forward(inputs))
 