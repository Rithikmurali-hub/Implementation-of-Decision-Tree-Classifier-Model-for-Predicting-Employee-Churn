# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score  

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rithik M
RegisterNumber: 212225040342 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy",random_state=100)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
sample=[[0.5,0.8,9,260,6,0,1,2]]
print("Prediction for sample:",dt.predict(sample))
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=x.columns,class_names=["stayed","left"],filled=True,rounded=True,fontsize=10)
plt.show()
```

## Output:
<img width="1047" height="347" alt="Screenshot 2026-03-10 103532" src="https://github.com/user-attachments/assets/cec26b9b-1047-4021-a327-6046072d8c3a" />
<img width="1040" height="359" alt="Screenshot 2026-03-10 103603" src="https://github.com/user-attachments/assets/cf63eabf-5ef8-42f5-9bc8-348f3b8db7ac" />
<img width="1043" height="549" alt="Screenshot 2026-03-10 103656" src="https://github.com/user-attachments/assets/c8012ffb-55fe-44e6-acd9-346f6bd4bae8" />
<img width="1042" height="385" alt="Screenshot 2026-03-10 103718" src="https://github.com/user-attachments/assets/7847f983-729d-40e7-9807-4abe159731c8" />
<img width="1046" height="618" alt="Screenshot 2026-03-10 103753" src="https://github.com/user-attachments/assets/632d236b-43a6-4fa6-83c1-8c9d7dad9ad9" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
