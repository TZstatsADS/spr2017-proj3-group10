# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:26:34 2017

@author: sh3559 Senyao Han
"""

import cv2, os
import pandas as pd
import numpy as np

# set the path
pic_path = 'C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data\rawdata'

# set the parameters used in Histogram of Oriented Gradient(Hog feature)
winSize = (128, 128)
block_size= (32, 32)
block_stride= (16, 16)
cell_size= (16, 16)
nbins = 9
padding = (16,16)
# Create hog feature extraction
hog = cv2.HOGDescriptor(winSize, block_size, block_stride, cell_size, nbins)



# Reading the images and calculating the features
hog_dict = {}
for pic in os.listdir(pic_path):
	if pic.split('.')[1] == 'jpg':
         pic_name = pic.split('.')[0]
        pic_n = pic_name.split('_')[1]
        img_read = cv2.imread(os.path.join(pic_path,pic))
        img_read = cv2.resize(img_read, (128,128)) # to get the same number of features
        hog_dict[pic_n] = hog.compute(img_read, padding)
		
my_dictionary = {k: v.tolist() for k, v in hog_dict.items()}
hog_feature = {k: [i[0] for i in v] for k, v in my_dictionary.items()}
hog_feature = pd.DataFrame.from_dict(hog_feature, orient='index')
# sort the dateframe
hog_feature_sort=hog_feature.sort_values(by=[0])
# Saving the contents into a file
hog_feature.to_csv('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data\hog_features.csv')

