"""
Implements different types of layers
- Allows for easy grouping of different types of activation
  functions
- Types:
	- Sigmoid
"""
import theano.tensor as T
import numpy         as np
from theano.tensor.shared_randomstreams import RandomStreams

theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))

class Sigmoid(object):
	def __init__(self,size):
		self.size = size
		self.activation = T.nnet.sigmoid
	def activation_probability(self,W,bias,inputs):
		activation_score = T.dot(inputs,W) + bias
		activation_probs = self.activation(activation_score)
		return activation_probs
	def sample(self,W,bias,inputs):
		return theano_rng.binomial(
				size  = self.size,
				n     = 1,
				p     = self.activation_probability(W,bias,inputs),
				dtype = theano.config.floatX
			)

class Softmax(Sigmoid):
	def __init__(self,size):
		self.size = size
		self.activation = T.nnet.softmaxj
	def sample(self,W,bias,inputs):
		return self.activation_probability(W,bias,inputs)

class OneHotSoftmax(Softmax):
	def sample(self,W,bias,inputs):
		return theano_rng.multinomial(
				size  = self.size,
				n     = 1,
				pvals = self.activation_probability(W,bias,inputs),
				dtype = theano.config.floatX
			)
	
