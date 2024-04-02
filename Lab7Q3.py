import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read Excel file into DataFrame
df = pd.read_excel(r"C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")

# Filter rows with marks in specified range
df_filtered = df[df['output'].isin([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])]

# Round continuous values in 'output' column to the nearest integer
df_filtered['output'] = df_filtered['output'].round().astype(int)

# Split features and target columns
X = df_filtered.iloc[:, :-1]
y = df_filtered['output']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'CatBoost': CatBoostClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'Naïve Bayes': GaussianNB()
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

# Tabulate results
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)
