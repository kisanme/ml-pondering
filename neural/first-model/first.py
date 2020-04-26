# X: -1 0 1 2 3 4 
# Y: -2 1 4 7 10 13
# Relationship: 3x + 1

import tensorflow as tf
import numpy as np
from tensorflow import keras

# neural network model declaration
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# data for input and the expected results; we'll find the relationship using our ML Model
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# learn the relationship between xs and ys for number of epoch times refining the leanrt model
model.fit(xs, ys, epochs= 500)

# now time for making some prediction for given x 
print(model.predict([10.0]))

# the result should be 31 but it is nearly 31 not exactly
# reason being neural networks deal with probabilities not with certainities
