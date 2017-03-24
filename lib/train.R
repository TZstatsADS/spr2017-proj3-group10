train <- function(dat_train, label_train){
  
  ### Train a SVM using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("e1071")
  
  # Set as a dataframe
  trainset<-cbind(y=label_train,dat_train)
  n<-nrow(trainset)
  
  # pca of trainset 
  pca_train<-prcomp(trainset[,-1])
  # new trainset after pca
  trainset_pca<-data.frame(y=trainset[,1], pca_train$x)
  
  ### Train with kenal SVM with parameter c=5 and gamma=100 
  
  model<-svm(y~.,data=trainset_pca,cost=5,gamma=100,scale=F,kernel="radial")
  
  svm_train<-predict(model,trainset_pca[-1])
  accu=1-sum(svm_train!=trainset_pca$y)/n  
  return(list(fit=model, training_accuracy=accu, pca_fit=pca_train))
}