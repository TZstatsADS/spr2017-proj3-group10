
library("EBImage")
img_dir<-"C:/Users/sh355/Documents/GitHub/spr2017-proj3-group10/data/rawdata"
n_files <- length(list.files(img_dir))
files = list.files(img_dir)
im<-list()
for(i in 1:n_files){
  img <- readImage(paste(img_dir, files[i], sep='/'))
  im[[i]]<-resize(img, 128, 128)
}

setwd("C:/Users/sh355/Documents/GitHub/spr2017-proj3-group10/data2")
for (i in 1:n_files){
  if (i<10){
    writeImage(im[[i]], paste("image_000", i,".jpg", sep=""))
  } else if (i<100){
    writeImage(im[[i]], paste("image_00", i,".jpg", sep=""))
  } else if (i<1000){
    writeImage(im[[i]], paste("image_0", i,".jpg", sep=""))
  } else{
    writeImage(im[[i]], paste("image_", i,".jpg", sep=""))
  }
}

library(rPython)
setwd("C:/Users/sh355/Documents/GitHub/spr2017-proj3-group10/lib")
python.load("surf_features.py")
