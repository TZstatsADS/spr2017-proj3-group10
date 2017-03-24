test <- function(fit_train, dat_test, pca_train){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("e1071")
  
  # Transform testset into PCA
  testset_pca<-predict(pca_train, newdata = dat_test)
  
  pred <- predict(fit_train, newdata=testset_pca)

  return(pred)
}