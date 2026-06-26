import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
target_names = data.target_names
df=pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target
x=df.drop("target",axis=1)
y=df["target"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nPredicted values:",y_pred)
print("\nActual values:",y_test.values)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred,target_names=target_names)
cm=confusion_matrix(y_test,y_pred)

print("\nPredicted values with target names:")
for i in range(len(y_pred)):
    print(f"Sample {i+1}: Predicted={target_names[y_pred[i]]}, Actual={target_names[y_test.iloc[i]]}")

print("\nModel Evaluation metrics:")
print("\nAccuracy:",accuracy)
print("Classification Report:\n",cr)
print("Confusion Matrix:\n",cm)