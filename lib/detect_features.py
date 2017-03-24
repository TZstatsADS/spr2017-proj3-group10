import cv2
import graphlab as gl
from os import listdir
from os.path import isfile, join
import pandas as pd


def detect_features(filename, key):
    image = cv2.imread(join(resized_path, filename))
    print(image.shape)
    name = "image_{0:04d}".format(key)
    kp, des = sift.detectAndCompute(image, None)
    df[key-1]
    img_name.append(name)
    kp_resized.append(kp)
    des_resized.append(des)
    print("Detecting Features: 'image_{0:04d}.jpg'".format(key))

resized_path = '/Users/galen/Desktop/image_classification/data/img_resize'
#edge_path = '/Users/galen/Google Drive/Applied Data Science Projects/Proj3/spr2017-proj3-group10/data/edge_images'
#sobel_path = '/Users/galen/Google Drive/Applied Data Science Projects/Proj3/spr2017-proj3-group10/data/sobel_images'
sframe_path = '/Users/galen/Desktop/image_classification/data/sframe'

img_name = []
kp_resized = []
des_resized = []


sift = cv2.xfeatures2d.SIFT_create()

file_names = []
for file in listdir(resized_path):
    if not file.startswith('.') and isfile(join(resized_path, file)):
        file_names.append(file)

# try this - it should work
df = pd.DataFrame(data=None, index=file_names, columns=['img', 'key_points', 'descriptors'])
for i in range(0, 10): #len(file_names)):
    detect_features(file_names[i], i+1)

df = pd.DataFrame.from_dict(des_resized, orient='columns', dtype=None)

sa_kp_resize = gl.SArray(df)

# sa_des_resize = gl.SArray(des_resized)
# sf_resized = gl.SFrame({'sift_kp': sa_kp_resize, 'sift_des': sa_des_resize})
# sf_resized.save(join(sframe_path, 'resized_features'))

# sf_edge = gl.SFrame({'sift_kp': kp_edge, 'sift_des': des_edge})
# sf_edge.save(join(sframe_path, 'edge_features'))
#
# sf_sobel = gl.SFrame({'sift_kp': kp_sobel, 'sift_des': des_sobel})
# sf_sobel.save(join(sframe_path, 'sobel_features'))

#
# kp_resized[filename], des_resized[filename] = sift.detectAndCompute(imageresized, None)
# kp_edge[filename], des_edge[filename] = sift.detectAndCompute(imageedge, None)
# kp_sobel[filename], des_sobel[filename] = sift.detectAndCompute(imagesobel, None)

#file_names = [f for f in listdir(resized_path) if isfile(join(resized_path, f))]

# kp_edge = {}
# des_edge = {}
# kp_sobel = {}
# des_sobel = {}
