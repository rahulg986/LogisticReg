#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:54:25 2018

@author: rahul
"""
#Import Libraries

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from scipy import stats
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

#%matplotlib inline
data = pd.read_csv('/Users/rahul/Desktop/Prog/LogisticReg/data.csv')
#Check the head of ad_data
data.head()

data.info()
        
data.describe()

#Exploratory Data Analysis
data.hist(figsize=[10,10],color="teal")
#Create a histogram of the Age
sns.distplot(data['Age'], bins=20)
#Create a jointplot showing Area Income versus Age.
sns.jointplot(data['Area Income'], data['Age'])
#Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(data['Age'], data['Daily Time Spent on Site'], kind= 'kde')
#sns.jointplot(data['Daily Time Spent on Site'], data['Daily Internet Usage'])
sns.jointplot(data['Daily Time Spent on Site'], data['Daily Internet Usage'])
#Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(data, hue='Clicked on Ad')


#Logistic Regression

#Split the data into training set and testing set using train_test_split

data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)
X = data.drop(['Clicked on Ad'], axis = 1)
print(X)
Y = data['Clicked on Ad']
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
model = LogisticRegression()

#Train and fit a logistic regression model on the training set.
model.fit(X_train, Y_train)

#Model Stats
#stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logitModel=sm.Logit(Y_train,X_train)
result=logitModel.fit()
print(result.summary())

#Predictions and Evaluations

#Now predict values for the testing data.

predictions = model.predict(X_test)
print(predictions)

# Making the Confusion Matrix 
confMatrix = confusion_matrix(Y_test, predictions)
confMatrix

#Create a classification report for the model.
print(classification_report(Y_test, predictions))