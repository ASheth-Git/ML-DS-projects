#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:56:10 2025

@author: alpesh
"""

import numpy as np

nn_architecture =  [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

print(type(nn_architecture))#class list
print(nn_architecture[0])# <class 'list'>
print(len(nn_architecture))


#initilization of layer

def init_layers(nn_architecture, seed = 99):
    #definition of initilization of layer
    #Input is list of dictionary
    #Seed for random number generator
    
    np.random.seed(seed)
    
    number_of_layers = len(nn_architecture)
    #number of layer here is 5
    
    params_values = {}
    #parameter container 
    #empty dictionary initilization


    for idx, layer in enumerate(nn_architecture):
    # Loop through each layer in the architecture list

        layer_idx = idx + 1
        # Layer index starts from 1 (instead of 0)
    
        layer_input_size = layer["input_dim"]
        # Get the number of input units for this layer
    
        layer_output_size = layer["output_dim"]
        # Get the number of output units for this layer
    
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        # Initialize the weight matrix W for this layer with small random values
        # Shape: (output_size, input_size)
        
        #here np.random.randn(2, 1) gives a random matrix of size 2 x 1 
        #i.e. 2 rows and 1 column
    
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        # Initialize the bias vector b for this layer with small random values
        # Shape: (output_size, 1)
        # A column matrix
        
    return params_values
    
# The above loop populates the params_values dictionary with:
# W{1}, W{2}, ..., W{n} — weight matrices for each layer
# b{1}, b{2}, ..., b{n} — bias vectors for each layer
# where n is the total number of layers in the neural network.

# initilization of different activation functions "ReLU" "relu" and so on...

#Sigmoid
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

#Rectified Linear Unit
def relu(Z):
    return np.maximum(0,Z)

#Backward Sigmoid
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

#Backward ReLU
def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


# Using the activation functions

# Forward propagation for a single layer
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # Compute the linear part of the activation function
    
    Z_curr = np.dot(W_curr, A_prev) + b_curr
# Z^{current layer} = W^{current layer} * A_prev^{previous layer} + b^{current layer}

# A is activation function 

    # Select the activation function
    if activation is "relu":
        activation_func = relu
        # Use ReLU activation function
    elif activation is "sigmoid":
        activation_func = sigmoid
        # Use Sigmoid activation function
    else:
        raise Exception('Non-supported activation function')
        # Raise an error if the activation function is not supported


    # Return the activated output and the linear Z value (used in backprop)
    # A^{current_layer} = activation(Z^{current_layer})
    return activation_func(Z_curr), Z_curr


# Forward propagation for the entire network

def full_forward_propagation(X, params_values, nn_architecture):
    
    memory = {}
    # Initialize an empty dictionary to store intermediate values (A and Z)
    # These will be used later during backpropagation
    
    A_curr = X
    # Set the current activation to the input data X
    
    for idx, layer in enumerate(nn_architecture):
        # Loop through each layer in the architecture
        
        layer_idx = idx + 1
        # Layer index starts from 1 
        #(to match parameter keys like W1, b1, etc.)
        
        A_prev = A_curr
        # Store the activation from the previous layer 
        #(or input for the first layer)
        
        activ_function_curr = layer["activation"]
        # Get the activation function for the current layer
        
        W_curr = params_values["W" + str(layer_idx)]
        # Get the weight matrix for the current layer
        
        b_curr = params_values["b" + str(layer_idx)]
        # Get the bias vector for the current layer
        
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        # Perform forward propagation for the current layer
        
        memory["A" + str(idx)] = A_prev
        # Store the previous activation A_prev in memory 
        #(needed for gradients)
        
        memory["Z" + str(layer_idx)] = Z_curr
        # Store the linear output Z_curr in memory 
        #(needed for computing activation gradients)
       
    return A_curr, memory
    # Return the final output A from the last layer (used for prediction or loss)
    # Also return memory containing all intermediate values for backpropagation



#Calculating cost

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    # Number of examples (columns) in the batch
    
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    # Compute the cross-entropy cost for logistic regression / binary classification:
    # This measures how far the predicted probabilities Y_hat are from the true labels Y
    
    return np.squeeze(cost)
    # Remove any extra dimensions from the cost and return a scalar value



# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    
    Y_hat_ = convert_prob_into_class(Y_hat)
    # Convert predicted probabilities into class labels (e.g., 0 or 1)
    
    return (Y_hat_ == Y).all(axis=0).mean()
    # Compare predicted labels to true labels:
    # - Check equality element-wise, for all classes (using all(axis=0))
    # - Compute the mean accuracy over all examples in the batch



def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    
    # Number of examples in the batch
    m = A_prev.shape[1]
    
    # Select the correct backward activation function based on the activation used
    if activation is "relu":
        backward_activation_func = relu_backward
        # ReLU backward function computes gradient of ReLU wrt Z
        
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
        # Sigmoid backward function computes gradient of Sigmoid wrt Z
        
    else:
        raise Exception('Non-supported activation function')
        # Raise error if activation is unsupported
    
    # Calculate gradient of loss wrt Z (linear output before activation)
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # Gradient of loss wrt weights W_curr
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    
    # Gradient of loss wrt bias vector b_curr
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    
    # Gradient of loss wrt activation from previous layer A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    # Initialize dictionary to store gradients for all parameters
    
    # Number of examples in the batch
    m = Y.shape[1]
    
    # Reshape Y to match the shape of predictions Y_hat
    Y = Y.reshape(Y_hat.shape)
    
    # Initialize gradient of loss wrt activation of last layer A (output layer)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    
    # This comes from derivative of cross-entropy loss wrt predictions
    
    # Loop over layers in reverse order for backpropagation
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        
        # Layer indices start from 1, so adjust index
        layer_idx_curr = layer_idx_prev + 1
        
        # Get activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        # Current gradient of loss wrt activation
        
        # Retrieve stored values from forward pass needed for backprop
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        # Retrieve current layer parameters
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        # Compute gradients for current layer
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        # Store computed gradients in dictionary
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    # Return dictionary containing gradients for all layers
    return grads_values

# Updating parameters of values i.e. weight "W" and  and vector "b"

def update(params_values, grads_values, nn_architecture, learning_rate):
    
    # Loop through each layer in the neural network
    for layer_idx, layer in enumerate(nn_architecture):
        
        layer_idx += 1 
        
        # Update the weights for the current layer by subtracting
        # the gradient scaled by the learning rate
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        
        # Update the biases for the current layer similarly
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    # Return the updated parameters dictionary
    return params_values


#Training


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)
            
    return params_values
# def train(X, Y, nn_architecture, epochs, learning_rate):
#     params_values = init_layers(nn_architecture, 2)
#     cost_history = []
#     accuracy_history = []
    
#     for i in range(epochs):
#         Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
#         cost = get_cost_value(Y_hat, Y)
#         cost_history.append(cost)
#         accuracy = get_accuracy_value(Y_hat, Y)
#         accuracy_history.append(accuracy)
        
#         grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
#         params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
#     return params_values, cost_history, accuracy_history

#--------------Test---------------#

import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import regularizers

from sklearn.metrics import accuracy_score

# Set global style once (Spyder handles this well)
sns.set_style("whitegrid")

# Constants
N_SAMPLES = 1000
TEST_SIZE = 0.1

# Generate dataset
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# the function making up the graph of a dataset
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()
    plt.show()
        

# Show dataset plot
make_plot(X, y, "Dataset")

# Training
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 10000, 0.01)
# Prediction
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, nn_architecture)

# Accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f} - Alpesh".format(acc_test))


# Building a model
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=200, verbose=1)

# Predict probabilities
Y_test_prob = model.predict(X_test)

# Convert probabilities to binary predictions
Y_test_hat = (Y_test_prob > 0.5).astype(int).flatten()

# Calculate and print accuracy
acc_test = accuracy_score(y_test, Y_test_hat)
print("Test set accuracy: {:.2f} - Keras".format(acc_test))


# boundary of the graph
GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
# output directory (the folder must be created on the drive)
OUTPUT_DIR = "./binary_classification_vizualizations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

grid = np.mgrid[GRID_X_START:GRID_X_END:100j,GRID_X_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid


def callback_keras_plot(epoch, logs):
    plot_title = "Keras Model - It: {:05}".format(epoch)
    file_name = "keras_model_{:05}.png".format(epoch)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    prediction_probs = model.predict(grid_2d, batch_size=32, verbose=0)
    make_plot(X_test, y_test, plot_title, file_name=file_path, XX=XX, YY=YY, 
              preds=prediction_probs, dark=True)
    



# output directory (the folder must be created on the drive)
OUTPUT_DIR = "./binary_classification_vizualizations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Adding callback functions that they will run in every epoch
testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=callback_keras_plot)

# Building a model
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# ✅ Train model with callback
history = model.fit(X_train, y_train, epochs=200, verbose=1, callbacks=[testmodelcb])


prediction_probs = model.predict(grid_2d, batch_size=32, verbose=0)
make_plot(X_test, y_test, "Keras Model", file_name=None, XX=XX, YY=YY, preds=prediction_probs)

def callback_numpy_plot(index, params):
    plot_title = "NumPy Model - It: {:05}".format(index)
    file_name = "numpy_model_{:05}.png".format(index//50)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    prediction_probs, _ = full_forward_propagation(np.transpose(grid_2d), params, nn_architecture)
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
    make_plot(X_test, y_test, plot_title, file_name=file_path,
              XX=XX, YY=YY, preds=prediction_probs, dark=True)
    
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 10000, 0.01, False, callback_numpy_plot)


prediction_probs_numpy, _ = full_forward_propagation(np.transpose(grid_2d), params_values, nn_architecture)
prediction_probs_numpy = prediction_probs_numpy.reshape(prediction_probs_numpy.shape[1], 1)
make_plot(X_test, y_test, "NumPy Model", file_name=None, XX=XX, YY=YY, preds=prediction_probs_numpy)