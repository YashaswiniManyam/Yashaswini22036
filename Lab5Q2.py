import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from numpy.linalg import pinv

# Load the data
df = pd.read_excel("Lab Session1 Data.xlsx")
col=["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]
data_matrix = df[col]
A=data_matrix.values
C =df[["Payment (Rs)"]].values

# Dimensionality of the vector space
dimensionality = A.shape[1]

# Number of vectors
num_vectors = A.shape[0]

# Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)

# Pseudo-Inverse to find the cost of each product
pseudo_inverse_A = np.linalg.pinv(A)
X = np.dot(pseudo_inverse_A, C)

# Mark customers as RICH or POOR
df['Customer_Type'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')

# Encode 'Customer_Type' to numerical values
label_encoder = LabelEncoder()
df['Customer_Type'] = label_encoder.fit_transform(df['Customer_Type'])

# Train-test split for classification
X_classification = df[col].to_numpy()
y_classification = df['Customer_Type'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2)

# Train a classifier model (e.g., Logistic Regression)
classifier_model = LinearRegression()
classifier_model.fit(X_train, y_train)

# Predictions
y_pred = classifier_model.predict(X_test)

# Calculate classification metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Dimensionality of the vector space: {dimensionality}")
print(f"Number of vectors in the vector space: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")
print(f"Cost of each product (using Pseudo-Inverse):\n{X}")
print(f"Classification Metrics:\nMSE: {mse}\nRMSE: {rmse}\nMAE: {mae}\nR2 Score: {r2}")