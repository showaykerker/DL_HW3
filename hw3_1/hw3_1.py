import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, glob
import pickle
import time
from sklearn.model_selection import train_test_split
from model import encoder, decoder, n_latent

LR = 0.0001

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.84

test_condition = '0.8x_latent_loss_latent12'
name = time.strftime("%m%d-%H:%M:%S", time.localtime())
log_dir = './logs/' + test_condition + '_' + name + '/'

with open('../faces_32.pkl', 'rb') as f:
	data = pickle.load(f)

data = np.array(data, dtype=np.float32)
#print(data[0])

train_x, test_x = train_test_split(data, test_size=0.2) 
print('\nImages Loaded.')



xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
ys = tf.placeholder(tf.float32, [None, 32, 32, 3]) 

sample, mean, var, encoder_merge_list  = encoder(xs).get_zmv()
output, decoder_merge_list = decoder(sample).get_img()


dec_flatten = tf.reshape(output, [-1, 32*32*3])
ys_flatten = tf.reshape(ys, [-1, 32*32*3])

recons_loss = tf.reduce_sum(tf.squared_difference(dec_flatten, ys_flatten), 1)
latent_loss = - 0.5 * tf.reduce_sum(1 + 2 * var - tf.square(mean) - tf.exp(2.0 * var + 1e-10), 1) * 0.8
loss = tf.reduce_mean(recons_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

loss_merge_list=[
	tf.summary.image('original_img', ys),
	tf.summary.scalar('loss/total', loss),
	tf.summary.scalar('loss/reconstructed', tf.reduce_mean(recons_loss)),
	tf.summary.scalar('loss/latent', tf.reduce_mean(latent_loss)),
]

if __name__ == '__main__':
	
	n_train = 5000
	batch_size = 128


	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		merge_op = tf.summary.merge(encoder_merge_list + decoder_merge_list + loss_merge_list)
		#merge_gen = tf.summary.merge()
		writer_train = tf.summary.FileWriter(log_dir+'train', sess.graph, flush_secs=10)
		writer_gen   = tf.summary.FileWriter(log_dir+'gen', flush_secs=10)

		sess.run(init)

		# Training

		i = 0

		try:
			for i in range(1, n_train+1):
				idx = np.arange(0, len(data))
				num = np.random.shuffle(idx)
				idx = idx[:batch_size]

				train_data = np.array([data[j] for j in idx])
				dict_ = {xs: train_data, ys: train_data}
				summary, _, l = sess.run([merge_op, optimizer, loss], feed_dict=dict_)
				
				writer_train.add_summary(summary, i)

				if i % 100 == 0 or i == 1:
					print('Step %i, Loss: %f' % (i, l))

				if i % 100 == 0 or i == 1:
					n_gen = 16
					r = [np.random.normal(0, 1, n_latent) for i in range(n_gen)]
					image = sess.run([output], feed_dict={sample: r})
					#writer_gen.add_summary(image, i)
					image = np.reshape(image, [-1, 32, 32, 3])
					image = np.multiply(image, 255)
					co = 0

					for img in image:
						path = log_dir+'gen/ep_'+str(i)+'/'
						if not os.path.exists(path): os.makedirs(path)
						cv2.imwrite(path+str(co)+'.png', img)
						co +=1 

					length = int(n_gen**0.5)
					big_pic = None
					for x_axis in range(length):
						row = np.concatenate(image[x_axis*length:x_axis*length+length-1], axis=0)
						if big_pic is None: big_pic = row.copy()
						else: big_pic = np.concatenate([big_pic, row], axis=1)
					cv2.imwrite(path+'Concat.png', big_pic)


		except :
			print('\nStop Training.')


		# Generate
		n_gen = 100
		r = [np.random.normal(0, 1, n_latent) for j in range(n_gen)]
		image = sess.run([output], feed_dict={sample: r})
		image = np.reshape(image, [-1, 32, 32, 3])
		image = np.multiply(image, 255)
		co = 0
		for img in image:
			path = log_dir+'gen/ep_'+str(i)+'/'
			if not os.path.exists(path): os.makedirs(path)
			cv2.imwrite(path+str(co)+'.png', img)
			co +=1 

		length = int(n_gen**0.5)
		big_pic = None
		for x_axis in range(length):
			row = np.concatenate(image[x_axis*length:x_axis*length+length-1], axis=0)
			if big_pic is None: big_pic = row.copy()
			else: big_pic = np.concatenate([big_pic, row], axis=1)
		cv2.imwrite(path+'gen_Concat.png', big_pic)

	


		# Reconstruct
		show = 100
		length = int(show**0.5)
		idx = np.arange(0, len(data))
		num = np.random.shuffle(idx)
		idx = idx[:length]

		train_data = np.array([data[j] for j in idx])
		dict_ = {xs: train_data, ys: train_data}
		image, _, l = sess.run([output, optimizer, loss], feed_dict=dict_)

		big_pic = None
		for x_axis in range(length):
			row = np.concatenate(image[x_axis*length:x_axis*length+length-1], axis=0)
			if big_pic is None: big_pic = row.copy()
			else: big_pic = np.concatenate([big_pic, row], axis=1)
		cv2.imwrite(path+'QQ.png', big_pic)