import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel("C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

feature_1 = 'embed_0'
feature_2 = 'embed_1'
class_feature = df["output"].values

# Assign classes based on the payment amount
df['Class'] = np.where(class_feature > 3, 1, 0)

# Define colors for each class
colors = {0: 'blue', 1: 'red'}

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Class'] == 0][feature_1], df[df['Class'] == 0][feature_2], color=colors[0], label='Class 0 (Blue)')
plt.scatter(df[df['Class'] == 1][feature_1], df[df['Class'] == 1][feature_2], color=colors[1], label='Class 1 (Red)')

# Add labels and legend
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Scatter Plot of Training Data with Assigned Classes')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
