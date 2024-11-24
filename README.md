# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:  Sanjai M
RegisterNumber:  24901269
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess dataset
dataset = pd.read_csv('Placement_Data.csv')

# Drop unnecessary columns
dataset = dataset.drop(['sl_no', 'salary'], axis=1)

# Convert relevant columns to categorical and encode them
categorical_columns = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in categorical_columns:
    dataset[col] = dataset[col].astype('category').cat.codes

# Define features (X) and target (Y)
X = dataset.iloc[:, :-1].values  # All columns except 'status'
Y = dataset.iloc[:, -1].values   # 'status' column (target)

# Initialize random weights (theta)
theta = np.random.randn(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, X, Y):
    h = sigmoid(X.dot(theta))
    return -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))

# Gradient Descent
def gradient_descent(theta, X, Y, alpha, num_iterations):
    m = len(Y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - Y) / m
        theta -= alpha * gradient
    return theta

# Train the model
theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h > 0.5, 1, 0)
    return y_pred

# Evaluate model accuracy
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == Y)
print("Accuracy:", accuracy)

# Predict for new data
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1]])  # Example 1
y_prednew = predict(theta, xnew)
print("Prediction for new sample:", y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1]])  # Example 2
y_prednew = predict(theta, xnew)
print("Prediction for new sample:", y_prednew)

```

## Output:
![logistic regression using gradient descent](sam.png) ![image](https://github.com/user-attachments/assets/744f1623-1dd1-4a69-ad2f-160846009ac6)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

