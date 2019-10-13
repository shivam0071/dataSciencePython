import matplotlib.pyplot as plt
import numpy as np

X_axis = np.arange(-10., 10., 0.2)
# X_axis = np.arange(-100., 100., 1 )
#X_axis is an one-dimensional array with 100 elements
# begin from -10.0 to 10.0, with interval of 0.2 for each two elements

Y_axis = np.linspace(0, len(X_axis), len(X_axis))
#Y_axis is also an one-dimensional arry with 100 elements
# begin from 0 to 100

# 0 to 100 and make is such that is has 100 elements ....if 50 is given as 3rd param
# than 50 would have been the no of elements

# The numpy.arange() let you set your array by indicating start and
# stop value in the first two parameters, and an interval for the third
# parameter. The size of elements will vary depending on the interval you set.
#
# Another function, nmpy.linspace(), is the same as for the first
# two parameters in numpy.arrange(), but it allows you to determine the
# array size by passing a integer to the third parameter, so the interval in
# the array will vary depending on the number you set

# plt.plot(X_axis, Y_axis)  # the plot works like this....for each value of X plot y
# plt.show()


import math

def sigmoid(x):
  a = []
  for item in x:
    a.append(1 / (1 + math.exp(-item)))
  return a

#
# x = np.arange(-10., 10, 0.2)
sig = sigmoid(X_axis) # the plot works like this....for each value of X plot y
plt.plot(X_axis,sig)
plt.show()
print("Plotted")