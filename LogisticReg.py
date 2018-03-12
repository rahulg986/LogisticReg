#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:54:25 2018

@author: rahul
"""
#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#%matplotlib inline
ad_data = pd.read_csv('/Users/rahul/Desktop/Project/Logistic/advertising.csv')
#Check the head of ad_data
ad_data.head(3)

ad_data.info()

ad_data.describe()

#Exploratory Data Analysis
#Create a histogram of the Age
sns.distplot(ad_data['Age'], bins=20)
#Create a jointplot showing Area Income versus Age.
sns.jointplot(ad_data['Area Income'], ad_data['Age'])
#Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(ad_data['Age'], ad_data['Daily Time Spent on Site'], kind= 'kde')
#sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'])
sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'])
#Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data, hue='Clicked on Ad')


#Logistic Regression

#Split the data into training set and testing set using train_test_split

ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)
X = ad_data.drop(['Clicked on Ad'], axis = 1)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
logmodel = LogisticRegression()

#Train and fit a logistic regression model on the training set.
logmodel.fit(X_train, y_train)

#Predictions and Evaluations

#Now predict values for the testing data.

predictions = logmodel.predict(X_test)
predictions

#Create a classification report for the model.
print(classification_report(y_test, predictions))