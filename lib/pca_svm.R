# set working directory
setwd("/Users/xuxuanzi/Desktop/spr2017-proj3-group10/data/training_data/")
library(e1071)
# load data
load("/Users/xuxuanzi/Desktop/spr2017-proj3-group10/data/siftFeatures.RData")
label<-read.csv("labels.csv",header = T)
sift_features<-t(sift_features)
y<-as.factor(label$V1)

# set as a dataframe
dat<-data.frame(y,sift_features)
dim(dat)

# split the data into testset and trainset
indice<-sample(1:nrow(dat),size=500,replace = F)
testset<-dat[indice,] # Contains 500 observations
trainset<-dat[-indice,] # Contains 1500 observations

# PCA of trainset
pca_train<-prcomp(trainset[,-1])
prop_var_train<-(pca_train$sdev)^2/sum((pca_train$sdev)^2)

plot(cumsum(prop_var_train),xlab="Principal Component",
     type="l", main="Cumulative Proportion of Variance")

# New trainset after PCA
trainset_pca<-data.frame(y=trainset[,1], pca_train$x)

# Transform testset into PCA
testset_pca<-predict(pca_train, newdata = testset[,-1])
testset_pca<-cbind(y=testset[,1],testset_pca) 

# SVM prediction using dataset after PCA
model1<-svm(y~.,data=trainset_pca,cost=10,gamma=50,scale=F,kernel="radial")

#predict testset using trained svm model
svm.pred<-predict(model1, testset_pca[-1])
#test error
1-sum(svm.pred!=testset_pca$y)/500

# Training error
svm_train<-predict(model1,trainset_pca[-1])
1-sum(svm_train!=trainset_pca$y)/500