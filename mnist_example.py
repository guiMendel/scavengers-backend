import tensorflow as tf
from tensorflow import keras
import numpy as np

# Get the mnist dataset
mnist = keras.datasets.mnist

# Load it's values
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize x values
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)


# Create a simple sequential model
model = keras.models.Sequential()

# Flatten image data to a 1 dimensional array
model.add(keras.layers.Flatten())

# First hidden layer
model.add(keras.layers.Dense(128, activation=tf.nn.relu))

# Second layer
model.add(keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# Compile it
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train it
model.fit(x_train, y_train, epochs=3)

# Test it
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)

print(validation_loss, validation_accuracy)
