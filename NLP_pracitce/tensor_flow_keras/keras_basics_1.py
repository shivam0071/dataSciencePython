import numpy as np
from sklearn.datasets import load_iris # data set

iris = load_iris()
# print(iris.DESCR)

X = iris.data
        # - sepal length in cm
        # - sepal width in cm
        # - petal length in cm
        # - petal width in cm

y = iris.target

# 0, 1 ,2         one hot encoding
# - Iris - Setosa [1,0,0]
# - Iris - Versicolour[0,1,0]
# - Iris - Virginica [0,0,1]

from keras.utils import to_categorical
# make output as onehot encoding

y = to_categorical(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.preprocessing import MinMaxScaler
# Transforms features by scaling each feature to a given range.
# This estimator scales and translates each feature individually such
# that it is in the given range on the training set, i.e. between zero and one.

# example
# print(np.array([5,10,15,20])/20)
# array([0.25,0.50,0.75,1])

scaler_object = MinMaxScaler()
scaler_object.fit(X_train) # train, notice we didnt used X_test

# transform the Xtest and Xtrain
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

# print(scaled_X_test)
# [[ 0.52941176  0.36363636  0.64285714  0.45833333]


# Now buid the neural network with Keras


from keras.models import Sequential
# a bunch of sequences of layers
# layers: list of layers to add to the model.

from keras.layers import Dense

# Just your regular densely-connected NN layer.
# Dense implements the operation: output = activation(dot(input, kernel) + bias)
# where activation is the element-wise activation function passed as the activation
# argument, kernel is a weights matrix created by the layer, and bias is a bias
# vector created by the layer (only applicable if use_bias is True).
# Note: if the input to the layer has a rank greater than 2, then it is
# flattened prior to the initial dot product with kernel.


model = Sequential()
model.add(Dense(8, input_dim= 4, activation='relu'))
# Give how many neurons should be there
# activation function...etc
# input dimension = 4 as we have 4 features
model.add(Dense(8, input_dim= 4, activation='relu'))

model.add(Dense(3, input_dim= 4, activation='softmax'))  # we have 3 neurons as
# the output is 1,0,0 form so the output will be [0.2, 0.3, 0.5]

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

model.fit(scaled_X_train,y_train,epochs=150, verbose=2)
# no of times it should go over the inputs
# verbose for progress bars


print(model.predict(scaled_X_test)) # probability of 0,1 or 2 in terms of numbers
print(model.predict_classes(scaled_X_test)) # class only [0,1,2]

predictions = model.predict_classes(scaled_X_test)
y_test.argmax(axis=1) # back to class... giving us the index positions

print(y_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test.argmax(axis=1),predictions))
print(classification_report(y_test.argmax(axis=1),predictions))
print(accuracy_score(y_test.argmax(axis=1),predictions))



# SAVING the MODEL
model.save("myfirstmodel.h5")


# LOADING THE MODEL

from keras.models import load_model

# new_model = load_model("myfirstmodel.h5")