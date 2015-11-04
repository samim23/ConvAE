# Copyright (c) 2015 ev0, lautimothy
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import time
import random
import numpy as np
import theano as thn
import theano.tensor as tn
import theano.tensor.nnet.conv as conv
import matplotlib.pyplot as plt
import cPickle as cpkl
from theano import shared
from theano.tensor.signal.downsample import max_pool_2d as max_pool
from theano.tensor.signal.downsample import max_pool_2d_same_size as max_pool_same
from skimage.transform import downscale_local_mean as downsample
from copy import deepcopy
from mlp import *
from util import *


def centerDataset(data):

	new_data = np.transpose(data, (0, 3, 1, 2))
	N, k, m, n = new_data.shape
	new_data = new_data.reshape(N, np.prod(new_data.shape[1:]))
	x = tn.fmatrix('x')
	f = thn.function([], (x - tn.mean(x, axis=0)) / tn.var(x, axis=0), givens={x: shared(np.asarray(new_data, dtype='float32'))})
	scaled_data = f()
	# replace illegal values with 0.
	scaled_data[np.where(np.isnan(scaled_data))] = 0
	scaled_data = scaled_data.reshape(N, k, m, n)
	scaled_data = np.transpose(scaled_data, (0, 2, 3, 1))

	return scaled_data


def displayCov(data):
	"""
	Display the covariance matrix of the pixels
	across the image channels.

	Args:
	-----
		data: No_imgs x img_height x img_width x no_channels array.
	"""

	new_data, k = np.transpose(data, (0, 3, 1, 2)), data.shape[3]
	new_data = new_data.reshape(new_data.shape[0], np.prod(new_data.shape[1:]))
	zero_data = new_data - np.mean(new_data, axis=0)
	cov = np.cov(zero_data.T)

	N = cov.shape[0]
	f, sps = plt.subplots(k, k, sharex=True, sharey=True)
	for i in xrange(0, k):
		for j in xrange(0, k):
			y, x = (N * i) / k, (N * j) / k
			sps[i, j].imshow(cov[y : y + (N / k), x : x + (N / k)], aspect='auto')

	plt.show()


def fastConv2d(data, kernel, convtype='valid', stride=(1, 1)):
	"""
	Convolve data with the given kernel.

	Args:
	-----
		data: A N x l x m2 x n2 array.
		kernel: An k x l x m1 x n1 array.

	Returns:
	--------
		A N x k x m x n array representing the output.
	"""
	_data, _kernel =  np.asarray(data, dtype='float32'), np.asarray(kernel, dtype='float32')
	d = tn.ftensor4('d')
	k = tn.ftensor4('k')
	f = thn.function([], conv.conv2d(d, k, None, None, convtype, stride), givens={d: shared(_data), k: shared(_kernel)})
	return f()


def rot2d90(data, no_rots):
	"""
	Rotate the 2d planes in a 4d array by 90 degrees no_rots times.

	Args:
	-----
		data: A N x k x m x n array.
		no_rots: An integer repr. the no. rotations by 90 degrees.

	Returns:
	--------
		A N x k x m x n array with each m x n plane rotated.
	"""
	#Stack, cut & place, rotate, cut & place, break.
	N, k, m, n = data.shape
	result = data.reshape(N * k, m, n)
	result = np.transpose(result, (2, 1, 0))
	result = np.rot90(result, no_rots)
	result = np.transpose(result, (2, 1, 0))
	result = result.reshape(N, k, m, n)

	return result


class PoolLayer():
	"""
	A pooling layer.
	"""

	def __init__(self, factor, poolType='avg', decode=False):
		"""
		Initialize the pooling layer.

		Args:
		-----
			factor: Tuple repr. pooling factor.
			poolType: String repr. the pooling type i.e 'avg' or 'max'.
		"""
		self.type, self.factor, self.grad, self.decode = poolType, factor, None, decode


	def bprop(self, dEdo):
		"""
		Compute error gradients and return sum of error from output down 
		to this layer.

		Args:
		-----
			dEdo: A N x k x m2 x n2 array of errors from prev layers.

		Returns:
		--------
			A N x k x x m1 x m1 array of errors.
		"""
		if self.decode:
			if self.type == 'max':
				return np.kron(dEdo, np.ones(self.factor)) #Sum errors down
			else:
				# Downsample
		else:
			if self.type == 'max':
				return np.kron(dEdo, np.ones(self.factor)) * self.grad
			else:
				return np.kron(dEdo, np.ones(self.factor)) * (1.0 / np.prod(self.factor))
			

	def update(self, eps_w, eps_b, mu, l2, useRMSProp, RMSProp_decay, minsq_RMSProp):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps_w, eps_b: Learning rates for the weights and biases.
			mu: Momentum coefficient.
			l2: L2 Regularization coefficient.
			useRMSProp: Boolean indicating the use of RMSProp.
		"""
		pass #Nothing to do here :P


	def feedf(self, data):
		"""
		Pool features within a given receptive from the input data.

		Args:
		-----
			data: An N x k x m1 x n1 array of input plains.

		Returns:
		-------
			A N x k x m2 x n2 array of output plains.
		"""
		if self.unpoolInstead:
			if self.type == 'max':
				return np.kron(data, np.ones(self.factor))
			else:
				return np.kron(data, np.ones(self.factor)) * (1.0 / np.prod(self.factor)) #Upsample
		else:
			if self.type == 'max':
				_data = np.asarray(data, dtype='float32')
				x = tn.ftensor4('x')
				f = thn.function([], max_pool(x, self.factor), givens={x: shared(_data)})
				g = thn.function([], max_pool_same(x, self.factor)/x, givens={x: shared(_data + 0.0000000001)})
				self.grad = g()
				self.grad[np.where(np.isnan(self.grad))] = 0
				return f()
			else:
				return downsample(data, (1, 1, self.factor[0], self.factor[1]))


class ConvLayer():
	"""
	A Convolutional layer.
	"""

	def __init__(self, noKernels, channels, kernelSize, outputType='relu', stride=1, init_w=0.01, init_b=0, decode=False):
		"""
		Initialize convolutional layer.

		Args:
		-----
			eord: String repr. type of the layer - "encode" or "decode".
			noKernels: No. feature maps in layer.
			channels: No. input planes in layer or no. channels in input image.
			kernelSize: Tuple repr. the size of a kernel.
			stride: STRIDE MOTHERFUCKER!!! DO YOU SPEAK IT!!! Integer repr. convolutional stride.
			outputType: String repr. type of non-linear activation i.e 'relu', 'tanh' or 'sigmoid'.
			init_w: Std dev of initial weights drawn from a std Normal distro.
			init_b: Initial value of biases.
		"""
		self.decode = decode
		self.o_type = outputType
		self.init_w, self.init_b = init_w, init_b
		self.kernels = self.init_w * np.random.randn(noKernels, channels, kernelSize[0], kernelSize[1])
		self.bias = self.init_b * np.ones((noKernels, 1, 1))
		self.stride = stride, stride
		self.d_stride = np.zeros(self.stride)
		self.d_stride[0, 0] = 1
		self.v_w, self.dw_ms, self.v_b, self.db_ms = 0, 0, 0, 0


	def bprop(self, dEdo):
		"""
		Compute error gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dEdo: A N x k x m2 x n2 array of errors from prev layers.

		Returns:
		-------
			A N x l x m1 x n1 array of errors.
		"""
		if self.o_type == 'sigmoid':
			dEds = dEdo * sigmoid(self.maps + self.bias) * (1 - sigmoid(self.maps + self.bias))
		elif self.o_type == 'tanh':
			dEds = dEdo * sech2(self.maps + self.bias)
		else:
			dEds = dEdo * np.where((self.maps + self.bias) > 0, 1, 0)

		dEds = np.kron(dEds, self.d_stride) #TODO: Investigate stride effect.
		if self.stride[0] > 1:
			dEds = dEds[:, :, :-(self.stride[0] - 1), :-(self.stride[1] - 1)]

		self.dEdb = np.sum(np.sum(np.average(dEds, axis=0), axis=1), axis=1).reshape(self.bias.shape)

		#Correlate.
		xs, dEds = np.swapaxes(self.x, 0, 1), np.swapaxes(dEds, 0, 1)
		self.dEdw = fastConv2d(xs, rot2d90(dEds, 2)) / dEdo.shape[0]
		self.dEdw = np.swapaxes(self.dEdw, 0, 1)
		self.dEdw = rot2d90(self.dEdw, 2)

		#Correlate
		dEds, kernels = np.swapaxes(dEds, 0, 1), np.swapaxes(self.kernels, 0, 1)
		if self.decode:
			return fastConv2d(dEds, rot2d90(kernels, 2))
		else:
			return fastConv2d(dEds, rot2d90(kernels, 2), 'full')


	def update(self, eps_w, eps_b, mu, l2, useRMSProp, RMSProp_decay, minsq_RMSProp):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps_w, eps_b: Learning rates for the weights and biases.
			mu: Momentum coefficient.
			l2: L2 Regularization coefficient.
			useRMSProp: Boolean indicating the use of RMSProp.
			RMSProp_decay: Decay term for the squared average.
			minsq_RMSProp: Constant added to square-root of squared average. 
		"""
		if useRMSProp:
			self.dw_ms = (RMSProp_decay * self.dw_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdw))
			self.db_ms = (RMSProp_decay * self.db_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdb))
			self.dEdw = self.dEdw / (np.sqrt(self.dw_ms) + minsq_RMSProp)
			self.dEdb = self.dEdb / (np.sqrt(self.db_ms) + minsq_RMSProp)
			self.dEdw[np.where(np.isnan(self.dEdw))] = 0
			self.dEdb[np.where(np.isnan(self.dEdb))] = 0

		self.v_w = (mu * self.v_w) - (eps_w * self.dEdw) - (eps_w * l2 * self.kernels)
		self.v_b = (mu * self.v_b) - (eps_b * self.dEdb) - (eps_b * l2 * self.bias)
		self.kernels = self.kernels + self.v_w
		self.bias = self.bias + self.v_b


	def feedf(self, data):
		"""
		Return the non-linear result of convolving the input data with the
		weights in this layer.

		Args:
		-----
			data: An N x l x m1 x n1 array of input plains.

		Returns:
		-------
			A N x k x m2 x n2 array of output plains.
		"""
		self.x = data
		if self.decode:
			self.maps = fastConv2d(self.x, self.kernels, 'full', stride=self.stride)
		else:	
			self.maps = fastConv2d(self.x, self.kernels, stride=self.stride)

		if self.o_type == 'tanh':
			return np.tanh(self.maps + self.bias)
		elif self.o_type == 'sigmoid':
			return sigmoid(self.maps + self.bias)

		return relu(self.maps + self.bias) #TODO: Investigate relu unit effect.


class ConvAE():
	"""
	Convolutional Autoencoder class.
	"""

	def __init__(self, layers={}):
		"""
		Initialize network.

		Args:
		-----
			layers: Dict. of fully connected and convolutional layers arranged heirarchically.
		"""
		self.layers, self.div_x_shape, self.div_ind = [], None, 0
		if layers != {} and len(layers["decoder"]) == len(layers["encoder"]):
			self.layers = deepcopy(layers["decoder"]) + deepcopy(layers["encoder"])
			self.div_x_shape = None
			self.div_ind = len(layers["decoder"])


	def train(self, data, params):
		"""
		Train autoencoder to compress data using the given params.

		Args:
		-----
			data : A no_imgs x img_length x img_width x no_channels array of images.
			params: A dictionary of training parameters.
		"""

		#Notify fc of training.
		for layer in self.layers[0 : self.div_ind]:
			layer.train = True

		#Check the parameters

		print "Training network..."
		plt.ion()
		N, itrs, errors = train_data.shape[0], 0, []

		for epoch in xrange(params['epochs']):

			avg_errors = []

			start, stop = range(0, N, params['batch_size']), range(params['batch_size'], N, params['batch_size'])

			for i, j in zip(start, stop):
 
				error = self.reconstruct(data[i:j]) - data[i:j] #Can only use l2 norm.
				#TODO: Save reconstruction at some point.
				self.backprop(error)
				self.update(params, itrs)

				# display average reconstruction error
				# avg_error
				print '\r| Epoch: {:5d}  |  Iteration: {:8d}  |  Avg Reconst Error: {:.2f}|'.format(epoch, itrs, avg_error)
				if epoch != 0 and epoch % 100 == 0:
  					print '--------------------------------------------------------------------'
  				if params['view_kernels']:
  					self.displayKernels()

  				itrs = itrs + 1
  				avg_errors.append(avg_error)

  			i = start[-1]
  			error = self.reconstruct(data[i:]) - data[i:]
			self.backprop(error)
			self.update(params, itrs)

			# display average reconstruction error
			# avg_error
			print '\r| Epoch: {:5d}  |  Iteration: {:8d}  |  Avg Reconst Error: {:.2f} |'.format(epoch, itrs, avg_error)
			if epoch != 0 and epoch % 100 == 0:
  				print '---------------------------------------------------------------------'
  			if params['view_kernels']:
  				self.displayKernels()

  			itrs = itrs + 1
  			avg_errors.append(avg_error)

  			plt.figure(2)
  			plt.show()
  			errors.append(np.average(avg_errors))
  			plt.xlabel('Epochs')
  			plt.ylabel('Reconstruction Error')
  			plt.plot(range(epoch + 1), errors, '-g')
  			plt.axis([0, params['epochs'], 0, 1.00]) # TODO: Change Y-axis bound. Can't be beyond 255 on avg for img.
  			plt.draw() #Add fucking legend later.

  		#Stop fc dropout after training.
		for layer in self.layers[0 : self.div_ind]:
			layer.train = False
		plt.ioff()

  		#Also add training intervals to save architecture.
  		print "Saving model..."
  		#self.saveModel('mnist_cnn_1')

  		print "Done."


	def backprop(self, dE):
		"""
		Propagate the error gradients through the network.

		Args:
		-----
			dE: A no_imgs x k_classes array of error gradients.
		"""
		error = dE.T
		for layer in self.layers[0 : self.div_ind]:
			error = layer.bprop(error)


	def reconstruct(self, imgs):
		"""
		Reconstruct the imgs from codings.

		Args:
		-----
			imgs: A no_imgs x img_length x img_width x img_channels array.

		Returns:
		-------
			A no_imgs x img_length x img_width x img_channels array.
		"""
		#Transform 4d array into N, no.input_planes, img_length, img_width array
		data = np.transpose(imgs, (0, 3, 1, 2))

		for i in xrange(len(self.layers) - 1, - 1, -1):
			data = self.layers[i].feedf(data)

		return np.transpose(data, (0, 1, 2, 3))


	def update(self, params, i):
		"""
		Update the network weights.

		Args:
		-----
			params: Training parameters.
		"""
		fc, conv = params['fc'], params['conv']

		eps_w = epsilon_decay(fc['eps_w'], fc['eps_decay'], fc['eps_satr'], i, fc['eps_intvl'])
		eps_b = epsilon_decay(fc['eps_b'], fc['eps_decay'], fc['eps_satr'], i, fc['eps_intvl'])

		for layer in self.layers[0 : self.div_ind]:
			layer.update(eps_w, eps_b, fc['mu'], fc['l2'], fc['RMSProp'], fc['RMSProp_decay'], fc['minsq_RMSProp'])

		eps_w = epsilon_decay(conv['eps_w'], conv['eps_decay'], conv['eps_satr'], i, conv['eps_intvl'])
		eps_b = epsilon_decay(conv['eps_b'], conv['eps_decay'], conv['eps_satr'], i, conv['eps_intvl'])

		for layer in self.layers[self.div_ind:]:
			layer.update(eps_w, eps_b, conv['mu'], conv['l2'], conv['RMSProp'], conv['RMSProp_decay'], conv['minsq_RMSProp'])


	def saveModel(self, filename):
		"""
		Save the current network model in file filename.

		Args:
		-----
			filename: String repr. name of file.
		"""
		model = {
			'decoder': self.layers[0 : self.div_ind],
			'encoder': self.layers[self.div_ind :]
		}

		f = open(filename, 'w')
		cpkl.dump(model, f, 1)
		f.close()


	def loadModel(self, filename):
		"""
		Load an empty architecture with the network model
		saved in file filename.

		Args:
		-----
			filename: String repr. name of file.
		"""
		f = open(filename, 'r')
		model = cpkl.load(f)

		if model != {} and self.layers == []:
			self.layers = model["decoder"] + model["encoder"]
			self.div_x_shape = None
			self.div_ind = len(model["decoder"])

		f.close()


	def displayKernels(self):
		"""
		Displays the kernels in the first convolutional layer.
		"""
		plt.figure(1)

		kernels = self.layers[len(self.layers) - 1].kernels
		k, l, m, n = kernels.shape
		if l == 2 or l > 4:
			print "displayKernels() Error: Invalid number of channels."
			pass

		x = np.ceil(np.sqrt(k))
		y = np.ceil(k/x)

		for i in xrange(k):
			plt.subplot(x, y, i)
			kernel = np.transpose(kernels[i], (2, 1, 0))
			if kernel.shape[2] == 1:
				plt.imshow(kernel[:, :, 0], 'gray')
			else:
				plt.imshow(kernel)
			plt.axis('off')

		plt.draw()


def testMnist():
	"""
	Test cnn using the MNIST dataset.
	"""

	print "Loading MNIST images..."
	data = np.load('data/cnnMnist.npz')
	train_data = data['train_data'][0:1000].reshape(1000, 28, 28, 1)
	valid_data = data['valid_data'][0:1000].reshape(1000, 28, 28, 1)
	test_data = data['test_data'].reshape(10000, 28, 28, 1)
	train_label = data['train_label'][0:1000]
	valid_label = data['valid_label'][0:1000]
	test_label = data['test_label']

	print "Initializing network..."
	# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
	# With great power comes great responsibility, so use it wisely: DECODER MUST BE A REFLECTION OF ENCODER!
	layers = {
		"decoder":[
				ConvLayer(6, 1, (6, 6), decode=True),
				PoolLayer((2, 2)),
				ConvLayer(16, 6, (5, 5), decode=True),
				PoolLayer((2, 2))
			],
		"encoder":[
				PoolLayer((2, 2)),
				EncodeLayer(16, 6, (5,5), decode=False),
				PoolLayer((2, 2)), #Subsampling only!
				ConvLayer(6, 1, (6,6), decode=False) #Ensure result is an integer!
			]
	}

	params = {
		'epochs': 20,
		'batch_size': 500,
		'view_kernels': True,

		'fc':{
			'eps_w': 0.0007,
			'eps_b': 0.0007,
			'eps_decay': 9,
			'eps_intvl': 30,
			'eps_satr': 'inf',
			'mu': 0.7,
			'l2': 0.95,
			'RMSProp': True,
			'RMSProp_decay': 0.9,
			'minsq_RMSProp': 0.0
		},

		'conv': {
			'eps_w': 0.0007,
			'eps_b': 0.0007,
			'eps_decay': 9,
			'eps_intvl': 30,
			'eps_satr': 'inf',
			'mu': 0.7,
			'l2': 0.95,
			'RMSProp': True,
			'RMSProp_decay': 0.9,
			'minsq_RMSProp': 0
		}
	}

	cnn = Cnn(layers)
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, params)


def testCifar10():
	"""
	Test the cnn using the CIFAR-10 dataset.
	"""

	print "Loading CIFAR-10 images..."
	data = np.load('data/cifar10.npz')
	train_data = data['train_data'][0:49500]
	valid_data = data['train_data'][49500:50000]
	test_data = data['test_data']
	train_label = data['train_label'][0:49500]
	valid_label = data['train_label'][49500:50000]
	test_label = data['test_label']

	i = random.randint(0, 49500)
	pic = train_data[i]

	print "Centering images..."
	train_data = centerDataset(train_data)
	valid_data = centerDataset(valid_data)
	test_data = centerDataset(test_data)

	f, (a, b) = plt.subplots(1, 2)
	a.imshow(pic)
	a.axis('off')
	b.imshow(train_data[i])
	b.axis('off')
	plt.show()
	f.get_tight_layout()

	print "Displaying covariance matrix across channels..."
	displayCov(train_data[0:1000])

	print "Initializing network..."
	# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
	layers = {
		"fc":[
				PerceptronLayer(10, 64, 0.9, 'softmax'),
				PerceptronLayer(64, 64, 0.8, 'tanh')
			],
		"conv":[
				ConvLayer(64, 32, (5,5)),
				PoolLayer((2, 2), 'avg'),
				ConvLayer(32, 32, (5,5)),
				PoolLayer((2, 2), 'max'),
				ConvLayer(32, 3, (5,5), init_w=0.0001)
			]
	}

	params = {
		'epochs': 30,
		'batch_size': 128,
		'view_kernels': True,

		'fc':{
			'eps_w': 0.001,
			'eps_b': 0.002,
			'eps_decay': 9,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0.03
		},

		'conv': {
			'eps_w': 0.001,
			'eps_b': 0.002,
			'eps_decay': 9,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0.004
		}
	}

	#cnn = Cnn(layers)
	#cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, params)



if __name__ == '__main__':

	testMnist()
	#testCifar10()