import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel("C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

feature = 'embed_0'

feature_values = dataset[feature]

plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {feature}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

feature_mean = np.mean(feature_values)
feature_variance = np.var(feature_values)

print(f"Mean of {feature}: {feature_mean}")
print(f"Variance of {feature}: {feature_variance}")
