# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standred libraries in python for gradient descent.
2. Upload the datase and Read the given Dataset.
3.Declare the default values for linear regression. 
4. Predict the values of y.
5. Plot th graph respect to hours and score using scatter plot functions.


## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: P Shobika
RegisterNumber: 212221230096

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('ex1.txt',header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

#gradient
def computeCost(X,y,theta):
  m=len(y)#length of the training data
  h=X.dot(theta)#hypothesis
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err) #returning j

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) #call the function

def gradientDescent (X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history
  
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="Blue")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,8]),theta)*10000
print("For population = 80,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:
![image](https://user-images.githubusercontent.com/94508142/204134743-8f8006ab-55f1-4572-bd85-b341c6c87b72.png)
## Function
![image](https://user-images.githubusercontent.com/94508142/204134771-5b38303d-748f-4c49-96a6-deb13291dfde.png)
## Gradient Descent
![image](https://user-images.githubusercontent.com/94508142/204134800-e24c8d60-1eac-4ef3-bf6c-b7e595e1ae4f.png)
## Cost functio using Gradient Descent
![image](https://user-images.githubusercontent.com/94508142/204134832-fc7e05dd-836d-4001-b839-1bb2ec7d39da.png)
## Linear Regression
![image](https://user-images.githubusercontent.com/94508142/204134854-253fc0ce-6738-41a1-b0d7-8095acc6a61e.png)
## Profit Prediction for a population of 35000 :
![image](https://user-images.githubusercontent.com/94508142/204134881-97aec048-f913-4013-b451-f87fb30e1a92.png)

## Profit Prediction for a population of 70000 :
![image](https://user-images.githubusercontent.com/94508142/204134887-cedce9ea-3b20-4223-bc40-e857e3b527e9.png)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
