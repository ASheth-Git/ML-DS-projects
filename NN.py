#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:10:31 2025

@author: alpesh
"""

# Import TensorFlow library and alias it as tf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Access the built-in MNIST handwritten digits dataset from TensorFlow
mnist = tf.keras.datasets.mnist

# Load the MNIST dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)#60000 picture each of 28*28 pixel size
print(np.max(x_train))#maximum height of each pixel
print(x_train[0].shape)#shape of each picture

# Normalize pixel values to the range [0, 1] to improve model performance
x_train, x_test = x_train / 255.0, x_test / 255.0


#making Neural Network
n_input= 784 # input layer (28*28 pixels)

n_hidden1 = 512 #1st hidden layer

n_hidden2 = 256 #2nd hidden layer

n_hidden3 = 128 #3rd hidden layer

n_output = 10 #output layer (0-9) i.e. why 10

# Set the learning rate for the optimizer; 
# controls how big each weight update is
learning_rate = 1e-4  
# Equivalent to 0.0001 — a small, stable step size

# Define how many training steps (iterations) 
# to perform
n_iterations = 1000  
# The model will update its weights 1000 times

# Specify the number of samples processed 
# before updating the model
batch_size = 128  
# Each training step uses 128 training examples 
# at once

# Set the dropout rate for regularization to 
# prevent overfitting
drop_out = 0.5  
# Randomly disables 50% of neurons 
# during training to improve generalization

plt.imshow(x_train[7])
plt.show()

# 1. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),         # 28×28 → 784 vector
    tf.keras.layers.Dense(n_hidden1, activation='relu'),   # First hidden layer
    tf.keras.layers.Dropout(drop_out),                     # Dropout for regularization
    tf.keras.layers.Dense(n_hidden2, activation='relu'),   # Second hidden layer
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(n_hidden3, activation='relu'),   # Third hidden layer
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(n_output, activation='softmax')  # Output layer: probabilities over 10 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),                  # Adam optimizer with your lr=1e-4
    loss='sparse_categorical_crossentropy',                            # appropriate for integer labels
    metrics=['accuracy']                                               # track accuracy during training
)

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=15,                # e.g. 15 full passes through the data
    validation_split=0.1,     # reserve 10% of training data for validation
    shuffle=True              # shuffle data each epoch
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.show()

model.save('mnist_model.keras')


