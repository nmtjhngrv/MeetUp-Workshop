import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit (X_train,y_train)

y_tahmin_model=model.predict(X_test)

plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,model.predict(X_train),color='red')
plt.xlabel('Deneyim')
plt.ylabel('Maas')

plt.title('Deneyim & Maas-Egitimseti')
plt.show()

plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train,model.predict(X_train),color='red')
plt.xlabel('Deneyim')
plt.ylabel('Maas')

plt.title('Deneyim & Maas-Egitimseti')
plt.show()