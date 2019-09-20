"""
```
Fraud Detection

1. Explore the dataset/visualize
2. Decide which features are important
3. Perform Machine Learning
4. Test our model in the testing set
```
"""

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import random

df = pd.read_csv('creditcard.csv', low_memory=False)

""" 
Shuffle the order of the DataFrame's rows:
https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
"""

df = df.sample(frac=1).reset_index(drop=True)
df.head()

"""
What/how many are our fraudulent cases?. 
We will particionate our data set with pandas location feature, indicating so, which are going to be the fraudulent
and the non fraudulent cases (checking column Class)
"""

fraud = df.loc[df['Class'] == 1]
non_fraud = df.loc[df['Class'] == 0]
print(len(fraud))
print(len(non_fraud))

"""Some exploratory data analysis with matplotlib:"""

ax = fraud.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
non_fraud.plot.scatter(x='Amount', y='Class', color='Blue', label='Normal', ax=ax)
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount, color= "Red")
ax1.set_title('Fraud')
ax2.scatter(non_fraud.Time, non_fraud.Amount, color = "blue")
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# Starting with Machine Learning :)

# Linear Regression**

"""a. Colect all the columns except for the last one, and separate our prediction feature:"""

x = df.iloc[:, :-1]  # all our input data
y = df['Class']  # Class is what we are trying to predict

"""b. Split the training and the testing data (testing size 35%)"""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

# Handling with missing values:

X_train = X_train.dropna()
y_train = y_train.dropna()

"""Logistic Regression model"""

logistic = linear_model.LogisticRegression(C=1e5)
# C is our penalty term to incentivate and regulate against overfitting C=100000
logistic.fit(X_train,y_train)
print('score: ', logistic.score(X_test, y_test))

"""score 0.99"""