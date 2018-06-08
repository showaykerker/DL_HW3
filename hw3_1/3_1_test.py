import tensorflow as tf
import numpy as numpy
import dataset
import os, glob
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy

# ref. https://goo.gl/ivmNWs

img_size = int(96/2)
img_size_flatten = img_size * img_size * 3
img_shape = (img_size, img_size, 3)

img_path = '../faces/'
data = []



def main():
	n_data = len(glob.glob(os.path.join(img_path+'*')))
	for img_name in glob.glob(os.path.join(img_path+'*')):
		img = cv2.imread(img_name)
		img = scipy.misc.imresize(img, 0.5)
		data.append(img)
		print('\r%d/%d (%5.3f%%)'%(len(data), n_data, len(data)/n_data*100), end='')
		if len(data)==100: break

	print('\nImages Loaded.')

	train, test = train_test_split(data, test_size=0.2)
	#plot_images(train, 8)
	
	input_holder = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='input_holder')
	input_image = tf.reshape(input_holder, [-1, img_size, img_size, 3])
	
	

######################### Models #########################
def add_weights(shape, std=0.05):
	return tf.Variable(tf.truncated_normal(shape, stddev=std))

def add_biases(shape, std=0.05):
	return tf.random_normal(shape=shape, stddev=std)

def add_conv(input_, channel_input, filter_size, n_filters, use_pooling=True):
	shape = [filter_size, filter_size, channel_input, n_filters]
	weights = add_weights(shape=shape)
	biases = add_biases(shape=n_filters)
	layer = tf.nn.conv2d(input=input_, filter=weights, strides=[1,1,1,1], padding='SAME')
	layer += biases
	if use_pooling: layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	layer = tf.nn.relu(layer)
	return layer, weights

def add_flatten(layer):
	layer_shape = layer.get_shape()
	n_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, n_features])
	return layer_flat, n_features

def add_fc(input_, n_inputs, n_outputs, use_relu=True):
	w = add_weights(shape=[n_inputs, n_outputs])
	b = add_biases(shape=[n_outputs])
	layer = tf.matmul(input_, w) + b
	if use_relu: layer = tf.nn.relu(layer)
	return layer
##########################################################



def add_weights(shape, stddev=0.05):
	return tf.Variable(tf.truncated_normal(shape, stddev))

def add_biases(length):
	return tf.Variable(tf.random_normal(shape=[length]))

def plot_images(images_input, n_show=5):
	
	indices = random.sample(range(len(images_input)), min(len(images_input), n_show*n_show))
	images_ = [images_input[i] for i in indices]

	fig, axes = plt.subplots(n_show, n_show)
	fig.subplots_adjust(hspace=0, wspace=0)

	for i, ax in enumerate(axes.flat):
		ax.imshow(images_[i].reshape(img_size, img_size, 3))
		ax.set_xticks([]); ax.set_yticks([]);

	plt.show()


if __name__ == '__main__':
	main()