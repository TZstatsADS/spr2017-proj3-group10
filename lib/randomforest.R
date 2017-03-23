library(dplyr)
library(e1071)
library(caret)
library(randomForest)
setwd("~/Desktop/5243 ADS/proj3/spr2017-proj3-group10-master")


# load data
load("~/Desktop/5243 ADS/proj3/spr2017-proj3-group10-master/data/siftFeatures.RData")
sift<-sift_features
label<-read.csv("./data/labels.csv",as.is=T,header = T)
sift<-t(sift)
mydata<-cbind(label,sift)
mydata<-as.data.frame(mydata)

mydata[,1]<-as.factor(mydata[,1])
names<-c("labels",paste0('Feature',1:5000))
colnames(mydata)<-names


# Split data into testing and training
N<-sample(c(1:2000),200,replace = F)
test<-mydata[N,]
train<-mydata[-N,]
####################################################


# Use Random forest

output.forest <- randomForest(labels ~ ., ntree=700, mtry=70,xtest=test[,-1],ytest=test[,1], importance=T, data = train)


# View the forest results.
print(output.forest) 

# predict train
train$predicted <-output.forest$predicted

# Create Confusion Matrix
confusionMatrix(data=train$predicted,
                reference=train$labels,
                positive='1')
#train accuracy  0.7006

# predict train
test$predicted <-output.forest$test[[1]]

# Create Confusion Matrix
confusionMatrix(data=test$predicted,
                reference=test$labels,
                positive='1')
# test accuracy  0.765 
