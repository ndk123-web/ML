import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy import stats

model = LogisticRegression()

x = np.array([1,2,3,4,5,6,7,8,9,11,10,22,120,110,100]).reshape(-1,1)
y = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Actual X_TEST")
print(x_test)

print("Predicted values:")
print(y_pred)

print("Actual values:")
print(y_test)

plt.scatter(x,y,color='blue')
plt.ylabel('Labels')
plt.xlabel('Inputs(Features)')
plt.show()