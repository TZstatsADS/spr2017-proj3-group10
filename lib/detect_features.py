import cv2
import graphlab as gl
from os import listdir
from os.path import isfile, join
import numpy as np


def detect_features(filename, key):
    image = cv2.imread(join(edge_path, filename))
    sa = gl.SArray(image, dtype=np.ndarray)
    edge_array.append(sa)
    print("Saving Image Array: 'image_{0:04d}.jpg'".format(key))

edge_path = '/Users/galen/Desktop/image_classification/data/img_edge'
sframe_path = '/Users/galen/Desktop/image_classification/data/sframe'

img_dict = {}

file_names = []
for file in listdir(edge_path):
    if not file.startswith('.') and isfile(join(edge_path, file)):
        file_names.append(file)

edge_array = gl.SFrame()
for i in range(0, 10): #len(file_names)):
    detect_features(file_names[i], i+1)

edge_array.save(join(sframe_path, 'edge_array'))