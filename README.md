# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all necessary packages and dataset that you need to implement Logistic Regression.
2.Copy the actual dataset and remove fields which are unnecessary.
3.Then select dependent variable and independent variable from the dataset.
And perform Logistic Regression.
4.Print the values of confusion matrix, accuracy, and classification report to find whether the student is placed or not.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PADMASREE.P
RegisterNumber:  212225040293
*/

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

print(df.head())

df1 = df.copy()

df1 = df1.drop(['sl_no', 'salary'], axis=1)

print("\nNull Values:\n", df1.isnull().sum())

print("\nDuplicate Values:", df1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])

print("\nProcessed Dataset:\n", df1.head())

x = df1.iloc[:, :-1]
y = df1['status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("\nAccuracy Score:", accuracy)

print("\nConfusion Matrix:\n", confusion)

print("\nClassification Report:\n", cr)

from sklearn import metrics
import matplotlib.pyplot as plt

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Not Placed', 'Placed']
)

cm_display.plot()

plt.show()
```

## Output:
<img width="989" height="769" alt="Screenshot 2026-05-11 155858" src="https://github.com/user-attachments/assets/5f7fdb83-868c-41e6-9fb3-3ed837f0e417" />

<img width="1225" height="762" alt="Screenshot 2026-05-11 155912" src="https://github.com/user-attachments/assets/0377993a-57ec-48d1-9b3d-abad702480c0" />

<img width="1258" height="774" alt="Screenshot 2026-05-11 155943" src="https://github.com/user-attachments/assets/6b1142eb-da6b-4350-a2ca-94e559285e75" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
