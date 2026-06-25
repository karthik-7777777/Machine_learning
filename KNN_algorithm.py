import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
target_names = data.target_names
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target
# print(df.head())
x=df.drop("target",axis=1)
y=df["target"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


scores = []
temp=int(np.sqrt(len(x_train)))
if temp%2==0:
    temp+=1
start=max(1,temp-6)
end=min(len(x_train),temp+6)
for k in range(start,end,2):
    model=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(model,x_train,y_train,cv=5)
    scores.append(score.mean())
bestK=scores.index(max(scores))+(temp-6)
print(f"Best K value: {bestK}")


model=KNeighborsClassifier(n_neighbors=bestK,weights='uniform',metric='minkowski',p=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nPredicted values:",y_pred)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred,target_names=target_names)
cm=confusion_matrix(y_test,y_pred)

print("\nPredicted values with target names:")
for i in range(len(y_pred)):
    print(f"Sample {i+1}: Predicted={target_names[y_pred[i]]}, Actual={target_names[y_test.iloc[i]]}")

print("\nModel Evaluation metrics:")

print("Accuracy:",accuracy)
print("Classification Report:\n",cr)    
print("Confusion Matrix:\n",cm)

