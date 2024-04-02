import pandas as pd
import numpy as np
from math import log2

# Read the dataset
data = pd.read_excel(r"C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

# Filter rows where the label is in the specified values
valid_labels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
data = data[data['output'].isin(valid_labels)]

# Calculate entropy
def calculate_entropy(labels):
    entropy = 0
    total_count = len(labels)
    label_counts = labels.value_counts()
    for count in label_counts:
        probability = count / total_count
        entropy -= probability * log2(probability)
    return entropy

# Calculate information gain
def calculate_information_gain(data, feature):
    entropy_before = calculate_entropy(data['output'])
    unique_values = data[feature].unique()
    total_count = len(data)
    weighted_entropy_after = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        entropy_after = calculate_entropy(subset['output'])
        subset_count = len(subset)
        weighted_entropy_after += (subset_count / total_count) * entropy_after
    information_gain = entropy_before - weighted_entropy_after
    return information_gain

# Find the feature with the highest information gain
features = data.columns[:-1]  # Exclude the label column
information_gains = {}
for feature in features:
    information_gain = calculate_information_gain(data, feature)
    information_gains[feature] = information_gain

root_feature = max(information_gains, key=information_gains.get)
print("Root feature for the decision tree:", root_feature)
