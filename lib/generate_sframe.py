import graphlab as gl
from os.path import join
import pandas as pd
import numpy as np

# set paths
sframe_dir = '/Users/galen/Desktop/image_classification/data/sframe'
img_dir = '/Users/galen/Desktop/image_classification/data'
label_file = '/Users/galen/Desktop/image_classification/data/csv_labels/labels.csv'
sift_file = '/Users/galen/Desktop/image_classification/data/csv_features/sift_features.csv'

# resized image
path_resize = join(img_dir,'img_resize')
path_edge = join(img_dir, 'img_edge')
path_sobel = join(img_dir, 'img_sobel')

sf_resized = gl.image_analysis.load_images(path_resize, format='JPG', with_path=False, random_order=False)
sf_edge = gl.image_analysis.load_images(path_edge, format='JPG', with_path=False, random_order=False)
sf_sobel = gl.image_analysis.load_images(path_sobel, format='JPG', with_path=False, random_order=False)

# get labels
labels = gl.SFrame.read_csv(label_file)
labels = labels.apply(lambda x: 'chicken' if x['V1']==0 else 'dog')

# create data frame object from csv
df = pd.read_csv(sift_file, dtype=np.float32)
df = df.transpose()
sdf = df.values.tolist()

# create SFrame object
sift = gl.SFrame({'sift_features': sdf})
sift.add_columns([labels, sf_resized['image']], namelist=['label', 'image'])
sift = sift.add_row_number()
sift.save(join(sframe_dir, 'sift_features'))

# create SFrame objects for the edge and sobel image sets
edge = gl.SFrame()
edge.add_columns([labels, sf_edge['image']], namelist=['label', 'image'])
edge = edge.add_row_number()
edge.save(join(sframe_dir, 'edge_sf'))

sobel = gl.SFrame()
sobel.add_columns([labels, sf_sobel['image']], namelist=['label', 'image'])
sobel = sobel.add_row_number()
sobel.save(join(sframe_dir, 'sobel_sf'))
