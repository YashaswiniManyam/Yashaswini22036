import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import make_scorer, accuracy_score

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

# AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initial weights
weights = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Perceptron model
perceptron_model = Perceptron()

# Hyperparameter grid
param_grid = {
    'alpha': [0.01, 0.05, 0.1],
    'max_iter': [500, 1000, 1500],
    'tol': [1e-3, 1e-4, 1e-5],
    'eta0': [0.1, 0.2, 0.3],
    'early_stopping': [True, False]
}

# Define scorer for accuracy
scorer = make_scorer(accuracy_score)

# RandomizedSearchCV
random_search = RandomizedSearchCV(perceptron_model, param_distributions=param_grid, n_iter=100, scoring=scorer, random_state=42, cv=5)

# Fit RandomizedSearchCV
random_search.fit(X, y)

# Get best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
