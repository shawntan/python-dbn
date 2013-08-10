import theano
import theano.tensor as T
import numpy         as np
from theano.tensor.shared_randomstreams import RandomStreams

class DBN(object):
	def __init__(self,layers):
		self.layers = layers
		self.rbms = [
				RBM(layer[i],layer[i+1])
				for i in xrange(len(layers)-1)
			]
	
	def fit(self,X,Y):
		trans_X = X
		for rbm in self.rbms:
			rbm.fit(trans_X)
			trans_X = rbm.t_transform(trans_X)
