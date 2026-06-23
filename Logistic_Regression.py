import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
target_names = data.target_names
print(data.target_names)
# print(df.head())
# print(df.info())
# df.to_csv('breast_cancer_data.csv', index=False)
from sklearn.model_selection import train_test_split
x = df.drop("target", axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted values:", y_pred)
for i in range(len(y_pred)):
    print(f"Predicted: {y_pred[i]}, Actual: {target_names[y_pred[i]]}")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
classification_report=classification_report(y_test, y_pred, target_names=target_names)
confusion_matrix=confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report)
print("Confusion Matrix:\n", confusion_matrix)