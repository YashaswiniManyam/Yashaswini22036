# CONTAINS CODES FROM QUESTIONS 4-9

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset
dataset = pd.read_excel("C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

# QUESTION 4

# Extract features (X) and target variable (y)
X = dataset.drop(columns=["output"]).values  # Features
y = dataset["output"].values  # Target variable
y = pd.cut(y, bins=3, labels=["low", "medium", "high"]) #Making y values discrete 


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# QUESTION 5

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)

# QUESTION 6

accuracy = neigh.score(X_test,y_test)
print("NN Accuracy (k = 3):",accuracy)

# QUESTION 7

y_pred = neigh.predict(X_test)

'''
# Clear representation of the report via scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel('ACTUAL VALUES')
plt.ylabel('PREDICTED VALUES')
plt.title('ACTUAL vs PREDICTED Values')
plt.show()
'''

test_index = 0  # Index of the test vector 
test_vect = X_test[test_index]

# QUESTION 8

neigh2 = KNeighborsClassifier(n_neighbors=1)
neigh2.fit(X_train, y_train)
nn_accuracy = neigh2.score(X_test, y_test)
print("NN Accuracy (k = 1):", nn_accuracy)
# Perform classification for the given test vector
predicted_class = neigh.predict([test_vect])
print("Predicted class for the test vector for NN accuracy (k = 3):", predicted_class)

# Vary k from 1 to 11 for the kNN classifier
k_values = list(range(1, 12))
accuracies = []

for k in k_values:
    kneigh = KNeighborsClassifier(n_neighbors=k)
    kneigh.fit(X_train, y_train)
    accuracy = kneigh.score(X_test, y_test)
    accuracies.append(accuracy)

# Plot the accuracy values for different values of k
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. k for kNN Classifier')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# QUESTION 9

# Predict classes for the training and test data using both classifiers
nn_train_predictions = neigh.predict(X_train)
nn_test_predictions = neigh.predict(X_test)
knn_train_predictions = neigh2.predict(X_train)
knn_test_predictions = neigh2.predict(X_test)

# Calculate confusion matrix and classification report for Nearest Neighbor (NN) classifier
nn_train_conf_matrix = confusion_matrix(y_train, nn_train_predictions)
nn_test_conf_matrix = confusion_matrix(y_test, nn_test_predictions)
nn_train_report = classification_report(y_train, nn_train_predictions)
nn_test_report = classification_report(y_test, nn_test_predictions)

# Calculate confusion matrix and classification report for kNN classifier with k = 3
knn_train_conf_matrix = confusion_matrix(y_train, knn_train_predictions)
knn_test_conf_matrix = confusion_matrix(y_test, knn_test_predictions)
knn_train_report = classification_report(y_train, knn_train_predictions)
knn_test_report = classification_report(y_test, knn_test_predictions)

# Print confusion matrices and classification reports
print("Confusion Matrix for Nearest Neighbor (NN) Classifier (Training Data):\n", nn_train_conf_matrix)
print("\nConfusion Matrix for Nearest Neighbor (NN) Classifier (Test Data):\n", nn_test_conf_matrix)
print("\nClassification Report for Nearest Neighbor (NN) Classifier (Training Data):\n", nn_train_report)
print("\nClassification Report for Nearest Neighbor (NN) Classifier (Test Data):\n", nn_test_report)
print("\nConfusion Matrix for kNN Classifier with k = 3 (Training Data):\n", knn_train_conf_matrix)
print("\nConfusion Matrix for kNN Classifier with k = 3 (Test Data):\n", knn_test_conf_matrix)
print("\nClassification Report for kNN Classifier with k = 3 (Training Data):\n", knn_train_report)
print("\nClassification Report for kNN Classifier with k = 3 (Test Data):\n", knn_test_report)
