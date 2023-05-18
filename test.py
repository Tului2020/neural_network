import numpy as np
import Layer


a_0 = [1]

# w_0 = [[1, 1]]
# b_1 = [1, 1]

# print(np.dot(a_0, w_0) + b_1)

layer1 = Layer.Layer_Dense(8)
print(layer1.weights, layer1.biases)
print(layer1.forward(a_0))



