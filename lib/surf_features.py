# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:17:43 2017

@author: sh3559 Senyao Han
"""

from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import os
from sklearn import preprocessing
import pandas


# Get all the path to the images and save them in a list
image_paths = []
for files in os.listdir('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data\rawdata'):
    dir = os.path.join('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data\rawdata', files)
    image_paths.append(dir)

# Create surf feature(Speeded-up Robust Features) extraction and keypoint detector objects
# Reading the image and calculating the features and corresponding descriptors
des_list = []
for  image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray, None)
    des_list.append((image_path, descs))  

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
# Perform k-means clustering
k = 5000 # Number of clusters
voc, variance = kmeans(descriptors, k, 1)  # Perform Kmeans with default values

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32") 
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1


# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# Calculating the number of occurrences
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Giving weight to one that occurs more frequently

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Saving the contents into a file
np.savetxt("r_surf_ft.csv",im_features,delimiter=",")
