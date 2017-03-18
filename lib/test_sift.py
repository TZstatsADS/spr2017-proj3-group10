#%%
import numpy as np
import cv2
import os
os.chdir('C:\\Users\sh355\Documents\GitHub\spr2017-proj3-group10\lib')
img = cv2.imread('image_0001.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
img=cv2.drawKeypoints(gray,kps, img)
cv2.imwrite('sift_keypoints.jpg',img)
descs_list = descs.tolist()
