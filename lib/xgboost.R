
library(xgboost)
library(dplyr)
library(e1071)
library(caret)

#load data
load("~/Desktop/5243 ADS/proj3/spr2017-proj3-group10-master/data/siftFeatures.RData")
sift<-sift_features
label <- rep(c(1,0), each = 1000)
sift<-t(sift)


# Split data into testing and training
N<-sample(c(1:2000),200,replace = F)
data_test<-sift[N,]
data_train<-sift[-N,]
lab_train<-label[-N]
lab_test<-label[N]
#############################################################

# use xgb

# train error 
# nthread = 2, bstSparse <- xgboost(data = data_train, label = lab_train, max.depth = 2, eta = 1, nround = 2, objective = "binary:logistic")

bstSparse <- xgboost(data = data_train, label = lab_train, max.depth = 2, eta = 0.19, nround = 2, nthread = 10,objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = data_train, label = lab_train)

bst <- xgboost(data = dtrain, max.depth = 2, eta = 0.19,  nround = 69,nthread = 10, objective = "binary:logistic", verbose = 0)

# test error
pred <- predict(bst, data_test)
confusionMatrix(data=as.numeric(pred > 0.5),
                reference=lab_test,
                positive='1')
