import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U

class Connections(object):
	"""
	Represents connections between n-groups input layers and
	one output layer
	"""
	def __init__(self,layers_in,layer_out):
		self.ins     = layers_in
		self.out     = layer_out
		self.Ws      = [ U.create_shared(U.initial_weights(inp.size,self.out.size))
							for inp in self.layers ]
		self.bias    = U.create_shared(np.zeros(outputs))
		self.updates = self.Ws + [self.bias]
	
	def transform(self,inputs):
		return self.out.mean(
				sum(T.dot(v,W) for v,W in zip(inputs,self.Ws)) +\
				self.bias
			)
	


