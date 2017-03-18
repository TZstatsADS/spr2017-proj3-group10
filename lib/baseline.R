sift<-read.csv("./sift_features/sift_features.csv",header = T)
label<-read.csv("labels.csv",as.is=T,header = T)
sift<-t(sift)
mydata<-cbind(label,sift)
N<-sample(c(1:2000),200,replace = F)
test<-mydata[N,]
train<-mydata[-N,]
gbm1 = gbm(V1 ~ .,data=train,distribution = "adaboost",
               n.trees = 200,interaction.depth = 1,
               bag.fraction = 0.5,train.fraction = 1,cv.folds = 5)

gbm.perf(gbm1)
#In my code I have used how many trees. 

library(dplyr)
preds <- predict(gbm1,as.data.frame(test),
                 n.trees=200,type="response")
density(preds) %>% plot
# Bandwidth

sum(test$V1!=as.numeric(preds> 0.5))/200
# error 




