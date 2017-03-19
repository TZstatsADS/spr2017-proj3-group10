#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Yuting Ma/Tian Zheng
### Project 3
### ADS Spring 2017

feature_up <- function(img_dir, set_name, export=T){
  
  ### Construct process features for training/testing images
  ### Sample simple feature: Extract row average raw pixel values as features
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains processed features for the images
  
  ### load libraries
  library("EBImage")
  
  n_files <- length(list.files(img_dir))
  files = list.files(img_dir)
  ### determine img dimensions
  img0 <-  readImage(paste(img_dir, files[1], sep='/'))
  mat1 <- as.matrix(img0)
  n_r <- nrow(img0)
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, n_r) 
  for(i in 1:n_files){
    img <- readImage(paste(img_dir, files[i], sep='/'))
    dat[i,] <- rowMeans(img)
  }
  
  ### output constructed features
  if(export){
    save(dat, file=paste0("../output/feature_", set_name, ".RData"))
  }
  return(dat)
}
