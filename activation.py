# import pprint
import numpy as np

# ReLU
def ReLU(x):
  return 0 if x < 0 else x

mapReLU = np.vectorize(ReLU)

# Layer
class Layer_Dense:
  def __init__(self, weights, biases) -> None:
    self.weights = np.array(weights)
    self.biases = np.array(biases)
  
  def forward(self, inputs):
    return mapReLU(np.dot(self.weights, inputs) + self.biases)

# Space Creation
# x = np.round(np.linspace(0, 1, 41), 2)
_x = [1]

# Layer 1
layer1 = Layer_Dense(
  [[1], [0]],
  [1, 0],
)
a1 = layer1.forward(_x)
print(a1)

# Layer 2
layer2 = Layer_Dense(
  [
    [1, 0], 
    [0, 1],
  ],
  [1, 0],
)
a2 = layer2.forward(a1)
print(len(a2))

# Layer 3
layer3 = Layer_Dense(
  [[1, 0]],
  [1],
)
aL = layer3.forward(a2)
print(aL)


