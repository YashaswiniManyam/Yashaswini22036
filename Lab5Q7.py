import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load your data from a CSV file
data = pd.read_excel("C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

# Assume your CSV file has columns named 'X1', 'X2', and 'label' (replace with your actual column names)
feature_1 = 'embed_0'
feature_2 = 'embed_1'

X = data[[feature_1, feature_2]].values
y = data["output"].values

# Discretize the continuous values into classes
y_class = pd.cut(y, bins=[-float('inf'), 1, 2, 3, 4, 5, float('inf')], labels=[0, 1, 2, 3, 4, 5], right=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Define the kNN classifier
knn_classifier = KNeighborsClassifier()

# Specify the hyperparameter grid to search
param_dist = {'n_neighbors': range(1, 20)}

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Parameters: ", random_search.best_params_)

# Get the best kNN model with the tuned hyperparameters
best_knn_model = random_search.best_estimator_

# Evaluate the model on the test set
accuracy = best_knn_model.score(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
