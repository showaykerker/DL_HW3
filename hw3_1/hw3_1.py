import tensorflow as tf
import numpy as np

import os, glob
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import pickle


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.84

# ref. https://goo.gl/ivmNWs

img_size = 48
img_size_flatten = img_size * img_size * 3
img_shape = (img_size, img_size, 3)

img_path = '../faces/'
data = []

loss_hist=[]
moving_average = 0.98



n_step = 600000
batch_size = 8


with open('../faces.pkl', 'rb') as f:
	data = pickle.load(f)



data = np.array(data, dtype=np.float32)
print(data[0])

train_x, test_x = train_test_split(data, test_size=0.2) 

print('\nImages Loaded.')

#plot_images(train, 8)

latent_size=2

input_holder = tf.placeholder(tf.float32, shape=[None, img_size_flatten])
input_img    = tf.reshape(input_holder, [-1, img_size, img_size, 3])
encoder_1    = tf.layers.conv2d(input_img, filters=16, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
encoder_1_2  = tf.layers.max_pooling2d(encoder_1, pool_size=2, strides=4)
encoder_2    = tf.layers.conv2d(encoder_1_2, filters=32, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
encoder_2_2  = tf.layers.max_pooling2d(encoder_2, pool_size=2, strides=4)
encoder_3    = tf.layers.flatten(encoder_2_2)
encoder_3_2  = tf.layers.dense(encoder_3, 128, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
encoder_3_2_2 = tf.layers.dropout(encoder_3_2, 0.8)
encoder_3_3  = tf.layers.dense(encoder_3_2_2, 32, activation=tf.nn.sigmoid, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())

z_mean = tf.layers.dense(encoder_3_3, latent_size, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
z_std  = tf.layers.dense(encoder_3_3, latent_size, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='eps')
z = z_mean + tf.exp(z_std/2)*eps



decoder_3_3 = tf.layers.dense(z, 32, activation=tf.nn.tanh, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.glorot_normal_initializer())
decoder_3_3_2 = tf.layers.dropout(decoder_3_3, 0.8)
decoder_3_2 = tf.layers.dense(decoder_3_3_2, 128, activation=tf.nn.tanh, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.glorot_normal_initializer())
decoder_3_2_2 = tf.layers.dropout(decoder_3_2, 0.8)
decoder_3_1 = tf.layers.dense(decoder_3_2, 288, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
decoder_3   = tf.reshape(decoder_3_1, shape=[-1, 3, 3, 32])
decoder_2_2 = tf.image.resize_bilinear(decoder_3, size=[12, 12], align_corners=None, name=None)
decoder_2   = tf.layers.conv2d_transpose(decoder_2_2, filters=16, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
decoder_1_2 = tf.image.resize_bilinear(decoder_2, size=[48, 48], align_corners=None, name=None)
decoder_1   = tf.layers.conv2d_transpose(decoder_1_2, filters=3, kernel_size=3, strides=(1,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
decoder_1_0 = tf.layers.flatten(decoder_1)
decoder_out = tf.layers.dense(decoder_1_0, 6912, activation=tf.nn.sigmoid, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.glorot_normal_initializer())

decoder_reshape = tf.reshape(decoder_out, shape=[-1, 48, 48, 3])

#decoder_3   = tf.reshape(decoder_3_2, shape=[-1, ])
'''
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(input_holder.get_shape())       # ?, 6912
	print(input_img.get_shape())          # ?, 48, 48, 3
	print(encoder_1.get_shape())          # ?, 48, 48, 16
	print(encoder_1_2.get_shape())        # ?, 12, 12, 16
	print(encoder_2.get_shape())          # ?, 12, 12, 32
	print(encoder_2_2.get_shape())        # ?,  3,  3, 32
	print(encoder_3.get_shape())          # ?, 288
	print(encoder_3_2.get_shape())        # ?, 128
	print(encoder_3_3.get_shape())        # ?, 32
	print(z_mean.get_shape())             # ?, 16
	print(z_std.get_shape())              # ?, 16
	print(z.get_shape())                  # ?, 16
	print(decoder_3_3.get_shape())        # ?, 32
	print(decoder_3_2.get_shape())        # ?, 128
	print(decoder_3_1.get_shape())        # ?, 288
	print(decoder_3.get_shape())          # ?,  3,  3, 32
	print(decoder_2_2.get_shape())        # ?, 12, 12, 32
	print(decoder_2.get_shape())          # ?, 12, 12, 16
	print(decoder_1_2.get_shape())        # ?, 48, 48, 32
	print(decoder_1.get_shape())          # ?, 48, 48,  3
	print(decoder_reshape.get_shape())    # ?, 48, 48,  3
	print(decoder_out.get_shape())        # ?, 6912

input()
'''


def plot_images(images_input, n_show=5):
	
	indices = random.sample(range(len(images_input)), min(len(images_input), n_show*n_show))
	images_ = [images_input[i] for i in indices]

	fig, axes = plt.subplots(n_show, n_show)
	fig.subplots_adjust(hspace=0, wspace=0)

	for i, ax in enumerate(axes.flat):
		ax.imshow(images_[i].reshape(img_size, img_size, 3))
		ax.set_xticks([]); ax.set_yticks([]);

	
	plt.show()


def Test(sess, n_show=8):
	inp=np.random.normal(4, 3, (64, latent_size))



	pics = sess.run(decoder_reshape, feed_dict={z:inp})

	for i, p in enumerate(pics):
		pics[i] = p*255
		#cv2.imshow('%d'%i, p)
	pics = np.array(pics, dtype=np.uint8)
	#print(inp)
	#print(pics[0], pics.shape)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	plot_images(pics, n_show=n_show)
	

def Forward(sess,x):
	print(np.reshape(x.copy(), [1, 6912]), np.reshape(x.copy(), [1, 6912]).shape)

	return sess.run(decoder_reshape, feed_dict={input_holder:np.reshape(x.copy(), [1, 6912])})



def VAE_loss(x_reconstructed, x_true):
	#encode_decode_loss = x_true * tf.log(1e-10+x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
	#encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
	encode_decode_loss = tf.losses.mean_squared_error(x_true, x_reconstructed)

	kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
	kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

	return tf.reduce_mean(encode_decode_loss + kl_div_loss)


loss_op = VAE_loss(decoder_out, input_holder)
#validate_op = VAE_loss(decoder_out, input_holder)
optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)



init = tf.global_variables_initializer()


plt.ion()

with tf.Session(config=config) as sess:
	sess.run(init)

	for i in range(1, n_step+1):
		idx = np.arange(0, len(data))
		num = np.random.shuffle(idx)
		idx = idx[:batch_size]
		train_data = np.array([data[i].flatten() for i in idx])


		feed_dict = {input_holder: train_data}
		_, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
		
		#loss_hist.append(l)
		if len(loss_hist) == 0: loss_hist.append(l)
		else: loss_hist.append(loss_hist[-1]*moving_average + l*(1-moving_average))

		if i % 200 == 0 or i == 1:
			print('Step %i, Loss: %f' %(i, l))

			'''
			#if l is np.nan:
			gen_pic = Forward(sess, train_data[0])

			print(type(gen_pic))
			print(gen_pic[0,...].shape)
			cv2.imshow('%d'%i, gen_pic[0,...])
			cv2.waitKey(1)
		
			'''
			plt.plot(loss_hist, 'g')
		
			plt.pause(0.0001)
			#Test(sess)
			#plt.show()

		if i % 1000 == 0:
			Test(sess)

		




	Test(sess)






