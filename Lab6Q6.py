import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron function
def perceptron(input_data, weights):
    # Calculate weighted sum
    weighted_sum = np.dot(input_data, weights)
    # Apply activation function
    return sigmoid(weighted_sum)

# Training data
X_train = np.array([
    [20, 6, 2, 386],   # C_1
    [16, 3, 6, 289],   # C_2
    [27, 6, 2, 393],   # C_3
    [19, 1, 2, 110],   # C_4
    [24, 4, 2, 280],   # C_5
    [22, 1, 5, 167],   # C_6
    [15, 4, 2, 271],   # C_7
    [18, 4, 2, 274],   # C_8
    [21, 1, 4, 148],   # C_9
    [16, 2, 4, 198]    # C_10
])

# Output labels (1 for high value, 0 for low value)
y_train = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Add bias term to input data
X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Initialize weights for perceptron learning
weights_perceptron = np.random.rand(X_train_bias.shape[1])

# Learning rate for perceptron learning
alpha_perceptron = 0.1

# Training the perceptron
for epoch in range(1000):  # Maximum 1000 epochs
    for i in range(len(X_train_bias)):
        # Make prediction
        prediction = perceptron(X_train_bias[i], weights_perceptron)
        # Calculate error
        error = y_train[i] - prediction
        # Update weights
        weights_perceptron += alpha_perceptron * error * X_train_bias[i]

# Matrix pseudo-inverse method
weights_pseudo_inv = np.linalg.pinv(X_train_bias) @ y_train

# Test data
X_test = np.array([
    [23, 5, 2, 350],   # Test data 1
    [17, 2, 4, 200]    # Test data 2
])

# Add bias term to test data
X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Predict using perceptron learning approach
predictions_perceptron = np.array([1 if np.dot(x, weights_perceptron) >= 0.5 else 0 for x in X_test_bias])

# Predict using matrix pseudo-inverse method
predictions_pseudo_inv = np.array([1 if np.dot(x, weights_pseudo_inv) >= 0.5 else 0 for x in X_test_bias])

# Compare results
for i in range(len(X_test)):
    print(f"Test data {i+1}: Perceptron = {predictions_perceptron[i]}, Pseudo-inverse = {predictions_pseudo_inv[i]}")
