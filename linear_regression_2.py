import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
data=housing.frame
print(data.head())
print(data.info())
data.to_csv('california_housing.csv', index=False)
x=data.drop('MedHouseVal',axis=1)
y=data['MedHouseVal']
# print(x.isnull().sum())
# print(y.isnull().sum())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predicted values:", y_pred)

print("\nCoefficients:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef}")

print("\nIntercept:", model.intercept_)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("\nModel Evaluation metrics:")
print("mean_absolute_error:",mean_absolute_error(y_test,y_pred))
print("mean_squared_error:",mean_squared_error(y_test,y_pred))
print("root_mean_squared_error:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2score:",r2_score(y_test,y_pred))

MedInc=float(input("Enter median income: "))
HouseAge=float(input("Enter house age: "))
AveRooms=float(input("Enter average rooms: "))
AveBedrms=float(input("Enter average bedrooms: "))
Population=float(input("Enter population: "))
AveOccup=float(input("Enter average occupancy: "))
Latitude=float(input("Enter latitude: "))
Longitude=float(input("Enter longitude: "))
new_data = pd.DataFrame({
    'MedInc': [MedInc],
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [AveBedrms],
    'Population': [Population],
    'AveOccup': [AveOccup],
    'Latitude': [Latitude],
    'Longitude': [Longitude]
})
new_prediction=model.predict(new_data)
print(f"predicted final price: {new_prediction[0]:.2f}")