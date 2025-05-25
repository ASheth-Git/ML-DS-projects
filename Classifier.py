#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 11:33:08 2025

@author: alpesh
"""

#Step0 importing useful library 

import sklearn 
#importing scikit learn

from sklearn.datasets import load_breast_cancer 
#loading data

from sklearn.model_selection import train_test_split 
#helps to split data into train and test

from sklearn.naive_bayes import GaussianNB
#importing the ML model name Naive Bayes
#useful for Binary Classification.

from sklearn.metrics import accuracy_score
#Evaluate accuracy of model


#Step 1 importing data set

data = load_breast_cancer()

# =============================================================================
# print(data.target_names)#check
# print(data.feature_names)#check
# =============================================================================

label_names=data['target_names']
labels=data['target']

# =============================================================================
# print(label_names)#check
# print(labels.shape)#check
# =============================================================================

features_names= data['feature_names']
features =data['data']

# =============================================================================
# print(features_names)#check 
# print(features)#check
# =============================================================================

#Step 2 Spliting data into training set and testing set

train,test,train_labels,test_labels=train_test_split(features,labels,test_size=0.50,random_state=42)

#Splits data into training and testing sets to evaluate model performance on unseen data.
# The function randomly splits data using test size which 
# is here 0.33 i.e. 33% of original data will be used to
# test. Plus rest of the 67% data will be used to train.
# We also have corresponding labels train_labels and 
# test_labels

#Step 4 Building and evaluating Model

#intitialize our classifier
gnb=GaussianNB()

#train our classifier
model=gnb.fit(train,train_labels)

# =============================================================================
# print(train)#check
# =============================================================================
#train is some data which is purely emprical (features)
# specifically data mean_radius, mean_texture ...etc

# =============================================================================
# print(train_labels)#check
# =============================================================================
#train_labels is some classified data meaning 
#it is 0's if malignant and 1's is benign (labels).

#make predictions
preds= gnb.predict(test)

# print(preds)
#Will return 0's and 1's

#step5 Evaluating the model's accuracy
print(accuracy_score(test_labels,preds))


