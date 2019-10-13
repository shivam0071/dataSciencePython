# X = -1 0 1 2 3 4
# Y = -3-1 1 3 5 7
#
# Relation?  Y = 2X - 1
# https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#3

# 28/09/2019  6:19 PM - Just another day spent alone
# Lets get familiar with the syntax and terminologies used

import tensorflow as tf
import numpy as np
from tensorflow import keras
# We need Keras
# Keras is a high-level neural networks API, written in Python and capable of
# running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on
# enabling fast experimentation. Being able to go from idea to result with the least
# possible delay is key to doing good research.
#
# Use Keras if you need a deep learning library that:
#
# Allows for easy and fast prototyping (through user friendliness,
# modularity, and extensibility).
# Supports both convolutional networks and recurrent networks,
#  as well as combinations of the two.
# Runs seamlessly on CPU and GPU.


# Creating a model

# what is sequential ? - The Sequential model is a linear stack of layers.
# The Sequential model API is a way of creating deep learning models where an
# instance of the Sequential class is created and model layers are created and added to it.
# The neural net used here is single layered - keras.layers.Dense - units = 1
# input_shape = [1] which means the input is single valued - X here  and predict Y

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# while compiling we gave 2 function..the loss and the optimizer
# So initially whne given X the model will make a guess...based on that guess the
# loss is calculateed ( the difference between prediction and Y)
# ...our aim is to minimize the loss and that is done by the optimizer
# the optimizer reduces the loss with each epochs

# SGD - https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
# mean_squared_error - https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Y = 3X + 1

# train it based on epochs
model.fit(xs, ys, epochs=200)

import pdb
pdb.set_trace()

print(model.predict([10.0])) # 31
#Actual [[31.00078]]