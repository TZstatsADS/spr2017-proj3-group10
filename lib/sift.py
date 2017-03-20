#%%
import imutils
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
#%%
import numpy as np
import cv2
import os
image_paths = []
for files in os.listdir('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data2'):
    dir = os.path.join('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\data2', files)
    image_paths.append(dir)

    #%%
des_list = []
for  image_path in image_paths:
    img = cv2.imread(image_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    des_list.append((image_path, descs))  
#%%      
descriptors = des_list[0][1]
for files, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
#%%
k = 500 
voc, variance = kmeans(descriptors, k, 1)
im_features = np.zeros((len(image_paths), k), "float32")
