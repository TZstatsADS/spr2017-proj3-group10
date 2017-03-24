import cv2
from os import listdir
from os.path import isfile, join


def process(filename, key):
    image = cv2.imread(join(rawimg_path,filename))
    print(image.shape)
    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] * r))
    imageresized = cv2.resize(image, (256, 256), dim, interpolation=cv2.INTER_AREA)
    imageedge = cv2.Canny(imageresized,100,200)
    imagesobel = cv2.Sobel(imageresized,cv2.CV_8U,0,1,ksize=5)

    cv2.imwrite(join(resized_path,'image_{0:04d}.jpg'.format(key)), imageresized)
    cv2.imwrite(join(edge_path, 'image_{0:04d}.jpg'.format(key)), imageedge)
    cv2.imwrite(join(sobel_path, 'image_{0:04d}.jpg'.format(key)), imagesobel)
    print('image_{0:04d}.jpg'.format(key))

# image path
rawimg_path = '/Users/galen/Desktop/image_classification/data/img_raw'
edge_path = '/Users/galen/Desktop/image_classification/data/img_edge'
resized_path = '/Users/galen/Desktop/image_classification/data/img_resize'
sobel_path = '/Users/galen/Desktop/image_classification/data/img_sobel'

file_names = [f for f in listdir(rawimg_path) if isfile(join(rawimg_path, f))]
for i in range(0, len(file_names)):
    process(file_names[i], i+1)