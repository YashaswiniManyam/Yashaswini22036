import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(input_data, weights):
    # Add bias term
    input_with_bias = np.insert(input_data, 0, 1)
    # Calculate weighted sum
    weighted_sum = np.dot(weights, input_with_bias)
    # Apply step activation function
    return step_activation(weighted_sum)

# XOR gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initial weights
weights = np.array([0.5, -0.5, 0.5])

# Learning rate
alpha = 0.1

# Maximum number of epochs
max_epochs = 1000

# List to store errors for plotting
errors = []

# Training
for epoch in range(max_epochs):
    error = 0
    for i in range(len(X)):
        # Make prediction
        prediction = perceptron(X[i], weights)
        # Calculate error
        error += (y[i] - prediction) ** 2
        # Update weights
        weights += alpha * (y[i] - prediction) * np.insert(X[i], 0, 1)
    # Append error for plotting
    errors.append(error)
    # Check for convergence
    if error == 0:
        print(f"Converged after {epoch+1} epochs")
        break

# Plotting epochs against error
plt.plot(range(1, epoch+2), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs for XOR Gate')
plt.show()

# Final weights
print("Final weights:", weights)

