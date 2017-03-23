library(gbm)
library(dplyr)
library(e1071)
library(caret)
setwd("~/Desktop/5243 ADS/proj3/spr2017-proj3-group10-master")

# load data
load("~/Desktop/5243 ADS/proj3/spr2017-proj3-group10-master/data/siftFeatures.RData")
sift<-sift_features
label<-read.csv("./data/labels.csv",as.is=T,header = T)
sift<-t(sift)
mydata<-cbind(label,sift)
names<-c("labels",paste0('Feature',1:5000))
colnames(mydata)<-names

# Split data into testing and training
N<-sample(c(1:2000),200,replace = F)
test<-mydata[N,]
train<-mydata[-N,]

####################################################

# use gbm

gbm1 = gbm(labels ~ .,data=train,distribution = "adaboost",
               n.trees = 512,shrinkage = 0.003,
               bag.fraction = 0.5,interaction.depth=3,train.fraction = 1,cv.folds = 5)


gbm.perf(gbm1,method = "cv")
#In my code I have used 512 trees. 

preds <- predict(gbm1,as.data.frame(test),
                 n.trees=512,type="response")
density(preds) %>% plot
# Bandwidth


# predict train
train$predicted <-predict(gbm1,as.data.frame(train),
                          n.trees=512,type="response")

# Create Confusion Matrix
confusionMatrix(data=as.numeric(train$predicted> 0.5),
                reference=train$labels,
                positive='1')
# train accuracy 0.7856

############################################################

# predict train
test$predicted <-predict(gbm1,as.data.frame(test),
                         n.trees=512,type="response")

# Create Confusion Matrix
confusionMatrix(data=as.numeric(test$predicted> 0.5),
                reference=test$labels,
                positive='1')

# test accuracy 0.73


