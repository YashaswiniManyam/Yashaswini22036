import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

# Perceptron function with different activation functions
def perceptron(input_data, weights, activation_function):
    # Add bias term
    input_with_bias = np.insert(input_data, 0, 1)
    # Calculate weighted sum
    weighted_sum = np.dot(weights, input_with_bias)
    # Apply activation function
    return activation_function(weighted_sum)

# XOR gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initial weights
weights = np.array([0.5, -0.5, 0.5])

# Learning rate
alpha = 0.1

# Maximum number of epochs
max_epochs = 1000

# Activation functions to iterate over
activation_functions = [bipolar_step_activation, sigmoid_activation, relu_activation]
activation_names = ['Bi-Polar Step', 'Sigmoid', 'ReLU']
convergence_epochs = []

# Training with different activation functions
for activation_function in activation_functions:
    # Reset weights for each activation function
    weights = np.array([0.5, -0.5, 0.5])
    for epoch in range(max_epochs):
        error = 0
        for i in range(len(X)):
            # Make prediction
            prediction = perceptron(X[i], weights, activation_function)
            # Calculate error
            error += (y[i] - prediction) ** 2
            # Update weights
            weights += alpha * (y[i] - prediction) * np.insert(X[i], 0, 1)
        # Check for convergence
        if error == 0:
            print(f"{activation_names[activation_functions.index(activation_function)]} converged after {epoch+1} epochs")
            convergence_epochs.append(epoch+1)
            break
    if error != 0:
        print(f"{activation_names[activation_functions.index(activation_function)]} did not converge after {max_epochs} epochs")

# Plotting convergence epochs for each activation function
if convergence_epochs:
    plt.bar(activation_names, convergence_epochs)
    plt.xlabel('Activation Function')
    plt.ylabel('Convergence Epochs')
    plt.title('Convergence Epochs for Different Activation Functions (XOR Gate)')
    plt.show()
else:
    print("None of the activation functions led to convergence.")
