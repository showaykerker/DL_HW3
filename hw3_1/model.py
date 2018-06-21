import tensorflow as tf
import numpy as np

CONV = {
	'e_conv1': { 'filters': 64, 'ksize':4, 'strides': 2, 'activation':tf.nn.relu },
	'e_conv2': { 'filters': 64, 'ksize':4, 'strides': 2, 'activation':tf.nn.relu },
	'e_conv3': { 'filters': 64, 'ksize':4, 'strides': 2, 'activation':tf.nn.relu },

	'd_conv_t_1': { 'filters': 64, 'ksize':4, 'strides': 2, 'activation':tf.nn.relu },
	'd_conv_t_2': { 'filters': 64, 'ksize':4, 'strides': 2, 'activation':tf.nn.relu },
}

n_latent = 8

FC = {
	'mean': { 'n_outputs': n_latent, 'activation':None },
	'var' : { 'n_outputs': n_latent, 'activation':None },

	'd_flat1': {'n_outputs': 6*6*3, 'activation':tf.nn.relu},
	'd_FC2'  : {'n_outputs': 32*32*3, 'activation':tf.nn.sigmoid}

}

class model_():
	def __init__(self):
		self.layers = []

	def __str__(self):

		ret = '<Class model>\n' 

		for info in self.layers:
			l = []
			s_k = list(info.keys())

			for k in sorted(s_k):
				l.append((k, info[k]))
			ret = ret + str(l) + '\n'

		ret = ret + '\n' + 'Total trainable: ' + str(self.count())

		return ret

	def count(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parameters = 1
			for dim in shape: variable_parameters *= dim.value
			total_parameters += variable_parameters
		return total_parameters

	def conv2d(self, input_, name, transpose=False):
		with tf.name_scope(name):
			regularizer = None #tf.contrib.layers.l2_regularizer(scale=0.1)
			if transpose:
				conv_t = tf.layers.conv2d_transpose(input_,
													filters     = CONV[name]['filters'], 
													kernel_size = CONV[name]['ksize'],
													strides     = CONV[name]['strides'], 
													padding     = 'SAME', 
													activation  = CONV[name]['activation'],)

				info = {'layer':conv_t, '0_name':'{:7s}'.format(name), '1_shape':'{:16s}'.format(str(conv_t.get_shape())), 'n_filters':CONV[name]['filters'], 'strides':CONV[name]['strides'], 'kernel_size':CONV[name]['ksize']}
			else:
				conv = tf.layers.conv2d(input_, 
										filters     = CONV[name]['filters'], 
										kernel_size = CONV[name]['ksize'],
										strides     = CONV[name]['strides'], 
										padding     = 'SAME', 
										activation  = CONV[name]['activation'],
										kernel_regularizer=regularizer)

				info = {'layer':conv, '0_name':'{:7s}'.format(name), '1_shape':'{:16s}'.format(str(conv.get_shape())), 'n_filters':CONV[name]['filters'], 'strides':CONV[name]['strides'], 'kernel_size':CONV[name]['ksize']}
			self.layers.append(info)

			return info['layer']

	def dense(self, input_, name):
		with tf.name_scope(name):
			shape = input_.get_shape().as_list()
			dim = 1
			for d in shape[1:]: dim *= d
			x = tf.reshape(input_, [-1, dim])
			ret = tf.layers.dense(x, FC[name]['n_outputs'], activation=FC[name]['activation'])			
			info = {'layer':ret, '0_name':'{:7s}'.format(name), 'n_output':FC[name]['n_outputs'] , '1_shape':'{:16s}'.format(str(ret.get_shape()))}
			self.layers.append(info)
			return info['layer']

	def flatten(self, input_, name):
		with tf.name_scope(name):
			ret = tf.contrib.layers.flatten(input_)
			info = {'layer': ret, '0_name':'{:7s}'.format(name), '1_shape':'{:16s}'.format(str(ret.get_shape()))}
			self.layers.append(info)
			return info['layer']


class encoder(model_):
	def __init__(self, X, ):
		super().__init__()
		with tf.name_scope('encoder'):
			self.layers.append({'0_name':'{:7s}'.format('Input'), '1_shape':'{:16s}'.format(str(X.get_shape()))})
			self.conv1 = self.conv2d(X, 'e_conv1')
			self.conv2 = self.conv2d(self.conv1 , 'e_conv2')
			self.conv3 = self.conv2d(self.conv2 , 'e_conv3')
			self.fl1   = self.flatten(self.conv3, 'flatten1')
			self.mean = self.dense(self.fl1, 'mean')
			self.var  = 0.5 * self.dense(self.fl1, 'var')
			self.eps  = tf.random_normal([tf.shape(self.fl1)[0], n_latent])
			self.z = self.mean + tf.exp(self.var + 1e-10)*self.eps
			
			self.merge_list=[
				tf.summary.histogram('encoder/e_conv1', self.conv1),
				tf.summary.histogram('encoder/e_conv2', self.conv2),
				tf.summary.histogram('encoder/e_conv3', self.conv3),
				tf.summary.histogram('encoder/mean', self.mean),
				tf.summary.histogram('encoder/var', self.var),
				tf.summary.histogram('encoder/z', self.z),
			]


	def get_zmv(self):

		return tf.reshape(self.z, shape=(-1, n_latent)), \
				tf.reshape(self.mean, shape=(-1, n_latent)),\
				tf.reshape(self.var, shape=(-1, n_latent)),\
				self.merge_list

class decoder(model_):
	def __init__(self, X, ):
		super().__init__()
		with tf.name_scope('decoder'):
			self.layers.append({'0_name':'{:7s}'.format('Input'), '1_shape':'{:16s}'.format(str(X.get_shape()))})
			self.flat = self.dense(X, 'd_flat1')
			self.reshape = tf.reshape(self.flat, [-1, 6, 6, 3])
			self.convt1 = self.conv2d(self.reshape, "d_conv_t_1", transpose=True)
			self.convt2 = self.conv2d(self.convt1,  "d_conv_t_2", transpose=True)
			self.flat2  = self.flatten(self.convt2, "d_flat2")
			self.dense2 = self.dense(self.flat2, "d_FC2")
			self.img = tf.reshape(self.dense2, [-1, 32, 32, 3])

			self.merge_list=[
				tf.summary.histogram('decoder/flat', self.flat),
				tf.summary.histogram('decoder/d_conv_t_1', self.convt1),
				tf.summary.histogram('decoder/d_conv_t_2', self.convt2),
				tf.summary.histogram('decoder/dense2', self.dense2),
				tf.summary.image('decoder/reconstructed', self.img),
			]


	def get_img(self):
		return self.img, self.merge_list



if __name__ == '__main__':
	X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	enc = encoder(X)
	z, m, v = enc.get_zmv()
	print(enc)

	X = tf.placeholder(tf.float32, shape=[None, 12])
	dec = decoder(X)
	img = dec.get_img()
	print(dec)

