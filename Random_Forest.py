import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# print(data.keys())

df=pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
x = df.drop('target', axis=1)
y=df['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50,max_features=int(np.sqrt(len(x.columns))), random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# print("Predictions:", y_pred)
# print("Actual:", y_test.values)

for i in range(len(y_pred)):
    print(f"Predicted: {data.target_names[y_pred[i]]}, Actual: {data.target_names[y_test.values[i]]}")

feature_importances = model.feature_importances_
feature_importances = pd.Series(feature_importances, index=x.columns).sort_values(ascending=False)
print("Feature Importances:", feature_importances)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
cr=classification_report(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)

print("Evaluation Metrics:")
print("Accuracy:", int(accuracy * 100),"%")
print("Classification Report:",cr)
print("Confusion Matrix:",cm)