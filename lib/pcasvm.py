#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:25:56 2017

@author: xuxuanzi
"""
# Import all packages used in this article
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

#import data
sift_feature= pd.read_csv("/Users/xuxuanzi/Desktop/spr2017-proj3-group10/data/training_data/sift_features/sift_features.csv")
sift_t = sift_feature.T
label = pd.read_csv("/Users/xuxuanzi/Desktop/spr2017-proj3-group10/data/training_data/labels.csv")

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


# Fit principal component analysis model to the data
pca = PCA()
pca.fit(X_train)
X_train_pca = pd.DataFrame(pca.transform(X_train))
X_test_pca = pd.DataFrame(pca.transform(X_test))

# Define training model based on nfolds cross-validatio
def svc_sel(Xtrain, ytrain, Xtest, ytest, nfolds):
    Cs = [5,10, 50,100,150]
    gammas = [40,50,100,150]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(Xtrain, ytrain)
    grid_search.best_params_
    grid_search.score(Xtrain,ytrain)
    grid_search.score(Xtest,ytest)
    return {'Best parameter': grid_search.best_params_, 
            'Training accurate rate': grid_search.score(Xtrain,ytrain),
            'Test accurate rate': grid_search.score(Xtest,ytest),
            'Best model': grid_search}
    
    
result = svc_sel(X_train_pca,y_train,X_test_pca,y_test,6)
result

# Predict one a new image based on training model
result['Best model'].predict(X_test_pca)
