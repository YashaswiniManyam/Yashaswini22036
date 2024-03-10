import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(input_data, weights, alpha):
    # Add bias term
    input_with_bias = np.insert(input_data, 0, 1)
    # Calculate weighted sum
    weighted_sum = np.dot(weights, input_with_bias)
    # Apply step activation function
    return step_activation(weighted_sum)

# AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initial weights
weights = np.array([10, 0.2, -0.75])

# Learning rates to iterate over
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Maximum number of epochs
max_epochs = 1000

# List to store convergence epochs for each learning rate
convergence_epochs = []

# Training for each learning rate
for alpha in learning_rates:
    # Reset weights for each learning rate
    weights = np.array([10, 0.2, -0.75])
    for epoch in range(max_epochs):
        error = 0
        for i in range(len(X)):
            # Make prediction
            prediction = perceptron(X[i], weights, alpha)
            # Calculate error
            error += (y[i] - prediction) ** 2
            # Update weights
            weights += alpha * (y[i] - prediction) * np.insert(X[i], 0, 1)
        # Check for convergence
        if error <= 0.002:
            print(f"Learning rate {alpha}: Converged after {epoch+1} epochs")
            convergence_epochs.append(epoch+1)
            break

# Plotting convergence epochs against learning rates
plt.plot(learning_rates, convergence_epochs, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Convergence Epochs')
plt.title('Convergence Epochs vs Learning Rate')
plt.grid(True)
plt.show()
