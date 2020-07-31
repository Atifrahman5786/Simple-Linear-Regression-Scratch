#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
from csv import reader
from matplotlib import pyplot as plt
import numpy as np


# In[86]:


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[87]:


filename = 'data.csv'
data = load_csv(filename)


# In[88]:


m = len(data)
X = []
Y = []
for i in range(m):
    X.append(float(data[i][0]))
    Y.append(float(data[i][1]))


# In[102]:


X[1:5]


# In[90]:


# scatter plot
plt.scatter(X, Y)
plt.xlabel('X_Axis')
plt.ylabel('Y_Axis')
plt.title('Scatter_Plot')
plt.tight_layout()
plt.show()


# In[91]:


theta = []
for i in range(2):
    theta.append(0)
theta    


# In[92]:


h = []
for i in range(m):
    h.append(0)


# In[93]:


#hypothesis function
def hypothesis(theta, X):
    for i in range(m):
        h[i] = theta[0] + (theta[1] * X[i])
    return h    


# In[94]:


# cost function
def cost_function(theta, X, Y):
    h = hypothesis(theta, X)
    error = list(np.array(h) - np.array(Y))
    error_sqr = [n ** 2 for n in error] 
    J = sum(error_sqr)/(2*m)
    return J


# In[95]:


J = cost_function(theta, X, Y)
print('With theta = 0 , J = ', J)
print('Expected cost function , J = 32.07')


# In[96]:


#set iterations and alpha
alpha = 0.01
iterations = 1500


# In[97]:


J_history = []
for i in range(iterations):
    J_history.append(0)


# In[98]:


def grad_descent(X, Y, theta, alpha, iterations):
    for i in range(iterations):
        h = hypothesis(theta, X)
        error = list(np.array(h) - np.array(Y))
        theta[0] = theta[0] - (alpha * sum(error)) / m
        theta1_change = []
        for i in range(m):
            err = error[i] * X[i]
            theta1_change.append(err)
        theta[1] = theta[1] - (alpha * sum(theta1_change)) / m   
        
        J_history[i] = cost_function(theta, X, Y)
    return theta        


# In[99]:


theta = grad_descent(X, Y, theta, alpha, iterations)


# In[100]:


print("Theta found by gradient descent : ", theta)
print("Expected theta value : [-3.63, 1.17]")


# In[101]:


#slope intercept form :- y = mx + b
X_ = np.array(X)
pred = theta[0] + theta[1] * X_
plt.scatter(X, Y)
plt.xlabel('X_Axis')
plt.ylabel('Y_Axis')
plt.title('Scatter_Plot')
plt.plot(X, pred, color = 'r')
plt.tight_layout()
plt.show()


# In[ ]:




