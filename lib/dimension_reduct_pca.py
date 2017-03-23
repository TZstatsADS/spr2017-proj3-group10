# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 04:50:37 2017

@author: sh3559 Senyao Han
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# load the data
sift_feature= pd.read_csv("C:/Users/sh355/Documents/GitHub/spr2017-proj3-group10/data/sift_features.csv")
sift_t = sift_feature.T
label = pd.read_csv("C:/Users/sh355/Documents/GitHub/spr2017-proj3-group10/data/labels.csv")

#make as a dataframe
dat = pd.concat([label.reset_index(drop=True), sift_t.reset_index(drop=True)],axis=1)
dat=dat.rename(columns = {'V1':'y'})

# Split data into trainset and test set
X_train, X_test, y_train, y_test = train_test_split(dat.drop('y',axis=1), dat['y'],
random_state=0)

# First, use linear SVM with grid search over a few choice of c to get a baseline
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
print("accuracy on training set: %f" % grid.score(X_train, y_train))
print("accuracy on test set: %f" % grid.score(X_test, y_test))

# preprocessing using zero mean and unit variance scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Fit principal component analysis model to the data
pca = PCA()
pca.fit(X_train)
X_train_pca = pd.DataFrame(pca.transform(X_train))
X_test_pca = pd.DataFrame(pca.transform(X_test))

# Saving the contents into a file
np.savetxt("X_train_pca.csv",X_train_pca,delimiter=",")
np.savetxt("X_test_pca.csv",X_test_pca,delimiter=",")