import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Assuming the training data is available in these variables
class0_x = np.random.uniform(low=1, high=10, size=(20,))
class0_y = np.random.uniform(low=1, high=10, size=(20,))
class1_x = np.random.uniform(low=1, high=10, size=(20,))
class1_y = np.random.uniform(low=1, high=10, size=(20,))

# Step 1: Generate test data points
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
test_points = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

# Step 2: Classify the test data points using kNN classifier
def classify_with_knn(k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(np.column_stack((np.concatenate((class0_x, class1_x)), np.concatenate((class0_y, class1_y)))), 
                       np.concatenate((np.zeros_like(class0_x), np.ones_like(class1_x))))
    predicted_labels = knn_classifier.predict(test_points)
    return predicted_labels

# Step 3: Make a scatter plot of the test data output
def plot_test_data(predicted_labels, k):
    plt.figure(figsize=(8, 6))
    plt.scatter(test_points[:, 0], test_points[:, 1], c=predicted_labels, cmap=plt.cm.coolwarm)
    plt.title(f'Test Data Output (k={k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Predicted Class')
    plt.grid(True)
    plt.show()

# Step 4: Repeat the process for various values of k and observe the class boundary lines
for k in [1, 3, 5, 9]:
    predicted_labels = classify_with_knn(k)
    plot_test_data(predicted_labels, k)
