import matplotlib.pyplot as plt
import numpy as np

# ReLU
def ReLU(x):
  return 0 if x < 0 else x
mapReLU = np.vectorize(ReLU)

# naming convention = [WEIGHT]_[LAYER_NUMBER]_[STARTING_NODE_NUMBER]_[ENDING_NODE_NUMBER]
# naming convention = [BIASE]_[LAYER_NUMBER]_[NODE_NUMBER]
w_1_s_1 = 6
w_1_s_2 = 0
b_1_1 = 0
b_1_2 = 0

w_2_1_1 = -1
w_2_1_2 = 0
w_2_2_1 = 0
w_2_2_2 = 0
b_2_1 = -0.7
b_2_2 = 0

w_L_1_L = -1
w_L_2_L = 0
b_L = 0

# Layer
class Layer_Dense:
  def __init__(self, weights, biases) -> None:
    self.weights = np.array(weights)
    self.biases = np.array(biases)
  
  def forward(self, inputs):
    return mapReLU(np.dot(self.weights, inputs) + self.biases)

# Space Creation
x = np.round(np.linspace(0, 1, 41), 2)
aL = []
for _x in x:
  _x = [_x]
  # Layer 1
  layer1 = Layer_Dense(
    [[w_1_s_1], [w_1_s_2]],
    [b_1_1, b_1_2],
  )
  a1 = layer1.forward(_x)

  # Layer 2
  layer2 = Layer_Dense(
    [
      [w_2_1_1, w_2_1_2],
      [w_2_2_1, w_2_2_2],
    ],
    [b_2_1, b_2_2],
  )
  a2 = layer2.forward(a1)

  # Layer 3
  layer3 = Layer_Dense(
    [[w_L_1_L, w_L_2_L]],
    [b_L],
  )
  aL.append(np.round(layer3.forward(a2)[0], 2))

plt.plot(x, aL)
plt.show()