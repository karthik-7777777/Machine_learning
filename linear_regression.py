import pandas as pd
import numpy as np
data=pd.read_csv('student_performance.csv')
import matplotlib.pyplot as plt
x=data.drop(['student_id','deep_work_sessions','assignment_completion_rate','social_media_hours','final_exam_score'],axis=1)
y=data['final_exam_score']
x.fillna(x.mean(), inplace=True)
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

# print(x.info())
# study_hours_per_day=float(input("Enter study hours per day: "))
# # deep_work_sessions=float(input("Enter deep work sessions: "))
# # assignment_completion_rate=float(input("Enter assignment completion rate: "))
# # social_media_hours=float(input("Enter social media hours: "))
# procrastination_index=float(input("Enter procrastination index: "))
# sleep_hours=float(input("Enter sleep hours: "))
# revision_efficiency=float(input("Enter revision efficiency: "))
# consistency_score=float(input("Enter consistency score: "))
# new_data = pd.DataFrame({
#     'study_hours_per_day': [study_hours_per_day],
#     # 'deep_work_sessions': [deep_work_sessions],
#     # 'assignment_completion_rate': [assignment_completion_rate],
#     # 'social_media_hours': [social_media_hours],
#     'procrastination_index': [procrastination_index],
#     'sleep_hours': [sleep_hours],
#     'revision_efficiency': [revision_efficiency],
#     'consistency_score': [consistency_score]
# })
# new_prediction=model.predict(new_data)
# print(f"predicted final score: {new_prediction[0]:.2f}")