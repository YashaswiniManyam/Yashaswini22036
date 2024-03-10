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
X = np.array([
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
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Initialize weights
weights = np.random.rand(4)

# Learning rate
alpha = 0.1

# Training the perceptron
for epoch in range(1000):  # Maximum 1000 epochs
    for i in range(len(X)):
        # Make prediction
        prediction = perceptron(X[i], weights)
        # Calculate error
        error = y[i] - prediction
        # Update weights
        weights += alpha * error * X[i]

# Test the perceptron on new data
test_data = np.array([
    [23, 5, 2, 350],   # Test data 1
    [17, 2, 4, 200]    # Test data 2
])

for data in test_data:
    prediction = perceptron(data, weights)
    if prediction >= 0.5:
        print(f"{data} is classified as High Value Transaction")
    else:
        print(f"{data} is classified as Low Value Transaction")
