import numpy as np

def NN(m1, w1, m2, w2, b):
  z = m1 * w1 + m2 * w2 + b
  print("Z is",z)
  return sigmoid(z)

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def cost_func(x):
  """Cost function is squared error"""
  # if we take w1 and w2 as 0 then cost would be
  # (b - 4) ** 2 where 2 is the actual output and b is the prediction (bias)
  return (x - 4) ** 2

def slope_cost_func(x):
  h = 0.000001
  return 2 * (x - 4) + h

def minimize_slope(x,epochs):

  num_iter = 0
  # while x > 1:
  for i in range(epochs):
    num_iter += 1
    print(f"EPOCHS {num_iter}: Bias {x}: Cost {cost_func(x)} : Slope {slope_cost_func(x)}")
    x = x - 0.1 * slope_cost_func(x)
    # num_iter += 1
  return x

if __name__ == "__main__":
  # 0 is for Blue flower and 1 for Red
  w1 = np.random.randn()
  w2 = np.random.randn()
  b = np.random.randn()

  # print(NN(5, w1, 3, w2, b))

  bias = 70
  print(cost_func(bias))
  # print(slope_cost_func(bias))
  minimize_slope(bias, 50)
  print("END")