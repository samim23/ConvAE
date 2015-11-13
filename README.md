ConvAE
======

A convolutional autoencoder for those times when you need to pretrain.


Dependencies
------------

You need the following dependencies if you choose to run the network:

* [Numpy] (http://www.numpy.org)
* [Skimage] (http://scikit-image.org)
* [Theano (Bleeding-edge)] (http://deeplearning.net/software/theano/install.html#install-bleeding-edge)


Initializing the network
----------------------

To initialize the network, pass a list containing each of the encoding layers (Convolutional and Pooling) arranged heirarchially - topmost layer at the beginning of list.

	layers = [
				PoolLayer((2, 2), 'max'),
				ConvLayer(16, 1, (3, 3))
			]

The network will automatically create the corresponding decoding layers.


Training
--------

Training the network is done by calling

	train(data, test, params)

**Arguments**:
	
* *data*: A no_imgs x img_length x img_width x no_channels array representing images.
* *test*: A no_imgs x img_length x img_width x no_channels array representing images.
* *params*: A dictionary of training parameters.

Here is a quick description of the training parameters:

* *epochs*: Integer repr. number of training epochs.
* *batch_size*: Integer repr. size of each training batch.
* *view_kernels*: Boolean indicating if kernels should be displayed.
* *view_recon*: Boolean indicationg if reconstructed images should be displayed.
* *no_images*: Integer repr. number of reconstructed images to display.
* *eps_w*: Integer repr. learning rate for weights/kernels.
* *eps_b*: Integer repr. learning rate for bias.
* *eps_decay*: Float repr. Decay constant for learning rate.
* *eps_intvl*: Integer repr. no. intervals to decay learning rate.
* *eps_satr*: Integer repr. no. iterations after which the learning rate stops decaying. If 'inf', learning rate keeps decaying throughout training.
* *mu*: Float repr. Momentum constant.
* *l2*: Float repr. l2 weight decay constant.
* *RMSProp*: Boolean option to use RMS prop.
* *RMSProp_decay*: Float repr. decay constant for RMS prop.
* *minsq_RMSProp*: Floar repr. constant for RMS prop denominator.

An example of training parameters is as follows
	
	{
		'epochs': 50,
		'batch_size': 500,
		'view_kernels': False,
		'view_recon': True,
		'no_images': 12,
		'eps_w': 0.005,
		'eps_b': 0.005,
		'eps_decay': 9,
		'eps_intvl': 10,
		'eps_satr': 'inf',
		'mu': 0.7,
		'l2': 0.95,
		'RMSProp': True,
		'RMSProp_decay': 0.9,
		'minsq_RMSProp': 0.01,
	}

Below shows both the reconstructed and actual images gotten from training the network on the Toronto Faces Dataset using the above parameters
![alt text](images/faces.png?raw=true "Faces images")


Loading and Saving models
-------------------------

You can save and load a trained model by calling `saveModel(filename)` and `loadModel(filename)` respectively.


Todo
----
1. Support greedy-layer wise training. Currently done manually by user. 
2. Investigate better error metric.
3. Implement the addition of noise during training.

