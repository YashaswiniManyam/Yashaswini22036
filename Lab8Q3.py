import pandas as pd
from math import log2

class DecisionTree:
    def __init__(self):
        self.tree = None
    
    def _calculate_entropy(self, labels):
        entropy = 0
        total_count = len(labels)
        label_counts = labels.value_counts()
        for count in label_counts:
            probability = count / total_count
            entropy -= probability * log2(probability)
        return entropy

    def _calculate_information_gain(self, data, feature):
        entropy_before = self._calculate_entropy(data['output'])
        unique_values = data[feature].unique()
        total_count = len(data)
        weighted_entropy_after = 0
        for value in unique_values:
          subset = data[data[feature] == value]
          if len(subset) == 0:  # Check if subset is empty
            continue
          entropy_after = self._calculate_entropy(subset['output'])
          subset_count = len(subset)
          weighted_entropy_after += (subset_count / total_count) * entropy_after
        information_gain = entropy_before - weighted_entropy_after
        return information_gain

    def _select_best_feature(self, data, features):
        information_gains = {}
        for feature in features:
            information_gain = self._calculate_information_gain(data, feature)
            information_gains[feature] = information_gain
        return max(information_gains, key=information_gains.get)
    
    def _build_tree(self, data, features):
        if len(data['output'].unique()) == 1:
            return data['output'].iloc[0]
        if len(features) == 0:
            return data['output'].mode()[0]
        best_feature = self._select_best_feature(data, features)
        tree = {best_feature: {}}
        remaining_features = [feat for feat in features if feat != best_feature]
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                tree[best_feature][value] = data['output'].mode()[0]
            else:
                subtree = self._build_tree(subset, remaining_features)
                tree[best_feature][value] = subtree
        return tree


    def fit(self, data):
        features = data.columns[:-1]
        self.tree = self._build_tree(data, features)

    def _predict_single(self, tree, instance):
        if isinstance(tree, dict):
            root = list(tree.keys())[0]
            subtree = tree[root]
            value = instance[root]
            if value in subtree:
                return self._predict_single(subtree[value], instance)
            else:
                return None
        else:
            return tree

    def predict(self, instances):
        predictions = []
        for idx, instance in instances.iterrows():
            prediction = self._predict_single(self.tree, instance)
            predictions.append(prediction)
        return predictions

# Test case
data = pd.read_excel(r"C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")
valid_labels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
data = data[data['output'].isin(valid_labels)]

# Create DecisionTree instance
dt = DecisionTree()

# Fit the model
dt.fit(data)

# Predict
predictions = dt.predict(data.iloc[:, :-1])
print("Predictions:", predictions)
