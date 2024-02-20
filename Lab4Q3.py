import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_excel("C:\\Users\\admin\\Desktop\\sem4\\training_mathbert 1.xlsx")

feature_1 = 'embed_0'
feature_2 = 'embed_1'

X = dataset[[feature_1, feature_2]].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

r_values = range(1, 11)
distances = []

for r in r_values:
    distance_r = np.linalg.norm(X_train[:, 0] - X_train[:, 1], ord=r)
    distances.append(distance_r)

plt.figure(figsize=(10, 6))
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
