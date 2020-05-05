###############
# Deep neural network implementation
###############

import tensorflow as tf
from tensorflow import keras

# load the fashion image data set
fashion_mnist = keras.datasets.fashion_mnist
# splits the data into train and test data sets
# labels are in numbers as well - to avoid bias on languages
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# build the model - with levels in the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    # last layer of neurons should have exactly the amount of classification classes
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model with the dataset
model.fit(train_images, train_labels, epochs=5)

# evaluate the prediction of our model with the accuracy it tests the test data
model.evaluate(test_images, test_labels)

# classify the total of test images
classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])