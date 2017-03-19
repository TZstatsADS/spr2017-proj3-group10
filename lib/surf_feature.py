import numpy as np
import cv2
import os
os.chdir('/Users/zixuan/Desktop/5243 ADS/proj3/training_data/raw_images/')
img = cv2.imread('image_1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
img=cv2.drawKeypoints(gray,kps, img)
cv2.imwrite('surf_keypoints.jpg',img)
descs_list = descs.tolist()
