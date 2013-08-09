import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))

def initial_weights(visible,hidden):
	"""
	return np.random.uniform(
			low  = -4 * np.sqrt(6./(visible+hidden)),
			high =  4 * np.sqrt(6./(visible+hidden)),
			size = (visible,hidden))
	"""
	return 0.1 * np.random.randn(visible,hidden)

def create_shared(array):
	return theano.shared(
			value = np.asarray(
				array,
				dtype = theano.config.floatX
			)
		)

