import tensorflow as tf
import numpy as np

import os, glob
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import pickle

import model

import time


tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.84

# ref. https://goo.gl/ivmNWs

img_size = 32
img_size_flatten = img_size * img_size * 3
img_shape = (img_size, img_size, 3)

img_path = '../faces/'
data = []

loss_hist=[]
moving_average = 0.98

test_condition = 'latent16_ll'
name = time.strftime("%m%d-%H:%M:%S", time.localtime())
log_dir = './logs/mod_w/' + test_condition + '_' + name + '/'


n_step = 600000
batch_size = 4


with open('../faces.pkl', 'rb') as f:
	data = pickle.load(f)



data = np.array(data, dtype=np.float32)
#print(data[0])

train_x, test_x = train_test_split(data, test_size=0.2) 

print('\nImages Loaded.')

#plot_images(train, 8)

latent_size=2

def encoder(X):
	conv1 = tf.layers.conv2d(X, )


def plot_images(images_input, n_show=5):
	
	indices = random.sample(range(len(images_input)), min(len(images_input), n_show*n_show))
	images_ = [images_input[i] for i in indices]

	fig, axes = plt.subplots(n_show, n_show)
	fig.subplots_adjust(hspace=0, wspace=0)

	for i, ax in enumerate(axes.flat):
		ax.imshow(images_[i].reshape(img_size, img_size, 3))
		ax.set_xticks([]); ax.set_yticks([]);

	plt.pause(0.001)
	


def Test(sess, n_show=8):
	'''
	for i in range(0, 10):
		print(Forward(sess, data[i]))
	'''
	inp = np.random.normal(0, 1, (64, latent_size))
	z_inp = np.random.rand()/8 + np.exp(np.random.rand()/16)*inp
	#z_inp = np.random.rand()/8 + np.random.rand()/8*inp

	pics = sess.run(decoder_reshape, feed_dict={z:z_inp})

	
	for i, p in enumerate(pics):
		pics[i] = p * 255
		if i < 3 : cv2.imshow('%d'%i, pics[i])
	pics = np.array(pics, dtype=np.uint8)
	#print(inp)
	#print(pics[0], pics.shape)
	cv2.waitKey(1)
	#cv2.destroyAllWindows()
	#@plot_images(pics, n_show=n_show)
	
	

def Forward(sess,x):
	print(np.reshape(x.copy(), [1, 6912]), np.reshape(x.copy(), [1, 6912]).shape)

	return sess.run(decoder_reshape, feed_dict={input_holder:np.reshape(x.copy(), [1, 6912])})



def VAE_loss(x_reconstructed, x_true):
	#encode_decode_loss = x_true * tf.log(1e-10+x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
	#encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
	encode_decode_loss = tf.reduce_sum(tf.losses.log_loss(x_true, x_reconstructed))

	kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
	kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

	w_rec = 1
	w_kl = 1

	loss = tf.reduce_sum(encode_decode_loss*w_rec + kl_div_loss*w_kl)

	tf.summary.scalar('loss/reconstruction_loss', tf.reduce_mean(encode_decode_loss)*w_rec)
	tf.summary.scalar('loss/kl_loss', kl_div_loss*w_kl)
	tf.summary.scalar('loss/total_loss', loss)

	return loss


loss_op = VAE_loss(decoder_out, input_holder)
#validate_op = VAE_loss(decoder_out, input_holder)
optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)



init = tf.global_variables_initializer()

plt.ion()



with tf.Session(config=config) as sess:
	# TensorBoard https://blog.csdn.net/sinat_33761963/article/details/62433234
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(log_dir, sess.graph)

	sess.run(init)

	for i in range(1, n_step+1):
		idx = np.arange(0, len(data))
		num = np.random.shuffle(idx)
		idx = idx[:batch_size]
		#print(idx)
		train_data = np.array([data[j].flatten() for j in idx])


		feed_dict = {input_holder: train_data}
		_, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
		tf.summary.scalar('loss', l)

		#loss_hist.append(l)
		if len(loss_hist) == 0: loss_hist.append(l)
		else: loss_hist.append(loss_hist[-1]*moving_average + l*(1-moving_average))

		if i % 10 == 0:
			summary = sess.run(merged, feed_dict=feed_dict)
			writer.add_summary(summary, i)

		if i % 200 == 0 or i == 1:
			print('Step %i, Loss: %f' %(i, l))
		
		if i % 50 == 0 : Test(sess)
			



			
			#plt.plot(loss_hist, 'g')
		
			#plt.pause(0.0001)
			#Test(sess)
			#plt.show()

		
			
			

	writer.close()
		




	Test(sess)






