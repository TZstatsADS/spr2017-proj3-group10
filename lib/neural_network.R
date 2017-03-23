# set working directory
setwd("/Users/xuxuanzi/Desktop/spr2017-proj3-group10/data/training_data/")
library(nnet)
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
trainset_pca<-data.frame(y=trainset[,1],pca_train$x)

# New testset
testset_pca<-data.frame(y=testset[,1],predict(pca_train, newdata = testset[,-1]))

# Neural Network Training
max_input = max(trainset_pca[,-1])
model1<-nnet(y ~ .,trainset_pca,size = 2,MaxNWts = 6000,maxit = 100,
             rang = 1/max_input,
             decay = 0)

# Training Accuracy
pred_train<-predict(model1,trainset_pca[,-1],type = "class")

table(pred_train,trainset_pca$y)
1-sum(pred_train!=trainset_pca$y)/1500

# Test Accuracy
pred_test<-predict(model1,testset_pca[,-1],type = "class")

table(pred_test,testset_pca$y)
1-sum(pred_test!=testset_pca$y)/500

