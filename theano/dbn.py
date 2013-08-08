import theano
import theano.tensor as T
import numpy         as np
from theano.tensor.shared_randomstreams import RandomStreams

class DBN(object):
	def __init__(self,layers):
		for i in xrange(1,len(layers)):
			assert(layers[i].input == layers[i-1].output)
		self.layers = layers
	
	def fit(self,X,Y):
		inter_X = X

		x = T.matrix('x')
		for l in self.layers[:-1]:
			l.fit(inter_X)
			inter_X   = theano.shared(np.asarray(inter_X,dtype=theano.config.floatX),borrow=True)
			curr_func = l.t_transform(inter_X)
			transform = theano.function(
					inputs  = []
					outputs = curr_func
				)
			inter_X = transform(inter_X)

			
		
			

