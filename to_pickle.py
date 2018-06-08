import pickle
import dataset
import os, glob
import cv2
import scipy.misc
import numpy as np
# ref. https://goo.gl/ivmNWs

img_size = int(96/2)
img_size_flatten = img_size * img_size * 3
img_shape = (img_size, img_size, 3)

img_path = 'faces/'
data = []




n_data = len(glob.glob(os.path.join(img_path+'*')))
for img_name in glob.glob(os.path.join(img_path+'*')):
	img = cv2.imread(img_name)
	img = scipy.misc.imresize(img, 0.5)
	data.append(img)
	print('\r%d/%d (%5.3f%%)'%(len(data), n_data, len(data)/n_data*100), end='')

print(data[0])

for i, d in enumerate(data):
	data[i] = (d/255)

print(data[0])

with open('faces.pkl', 'wb') as f:
	pickle.dump(data, f)

print('Done.')