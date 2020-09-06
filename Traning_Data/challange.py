import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#objectives
#-->Download
#-->Load
#-->Visualise
#-->Normalisation

#Load 
X = pd.read_csv('Traning_Data/Linear_X_Train.csv')
y = pd.read_csv('Traning_Data/Linear_Y_Train.csv') 
plt.scatter(X,y)

#calculate
lr=LinearRegression(normalize=True)
lr.fit(X,y)
print(lr.predict([[4]]))

plt.scatter(X,y)
plt.plot(X,lr.predict(X),color="orange")
plt.show

#joblib helps in saving the model to disk
joblib.dump(lr,"model.pkl")
m = joblib.load("model.pkl")

print(m.predict([[4]]))




