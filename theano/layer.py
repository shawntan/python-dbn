"""
Implements different types of layers
- Allows for easy grouping of different types of activation
  functions
- Types:
	- Sigmoid
"""
import theano.tensor as T
import numpy         as np
import utils,theano
from theano.tensor.shared_randomstreams import RandomStreams

theano_rng = utils.theano_rng
class Sigmoid(object):
	def __init__(self,size):
		self.size = size
		self.activation = T.nnet.sigmoid

	def activation_probability(self,W,bias,inputs):
		activation_score = T.dot(inputs,W) + bias
		activation_probs = self.activation(activation_score)
		return activation_score, activation_probs

	def sample(self,W,bias,inputs):
		activation_score,activation_probs = \
				self.activation_probability(W,bias,inputs)
		return \
			activation_score,\
			activation_probs,\
			theano_rng.binomial(
				size  = activation_probs.shape,
				n     = 1,
				p     = activation_probs,
				dtype = theano.config.floatX
			)

class Softmax(Sigmoid):
	def __init__(self,size):
		self.size = size
		self.activation = T.nnet.softmax
	def sample(self,W,bias,inputs):
		activation_score,activation_probs = \
				self.activation_probability(W,bias,inputs)
		return \
			activation_score,\
			activation_probs,\
			activation_probs

class OneHotSoftmax(Softmax):
	def sample(self,W,bias,inputs):
		activation_score,activation_probs = \
				self.activation_probability(W,bias,inputs)
		return \
			activation_score,\
			activation_probs,\
			theano_rng.multinomial(
				size  = activation_probs.shape,
				n     = 1,
				pvals = activation_probs,
				dtype = theano.config.floatX
			)
	
