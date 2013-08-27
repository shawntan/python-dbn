import theano.tensor as T
import numpy         as np
import utils         as U
import theano
from layers import Sigmoid
NO_UPDATES = theano.OrderedUpdates()
theano_rng = U.theano_rng

class Recurrent(Sigmoid):
	def __init__(self,size):
		super(Recurrent,self).__init__(size)
		self.W = U.create_shared(U.initial_weights(size,size))
		self.h0 = U.create_shared(np.zeros((size,)))
		self.updates = [self.W]
	def mean(self,activation_score):
		"""
		In using this class, the assumption is made
		that data provided is in increasing time order
		(I mean, how else would you arrange it?)
		"""
		def step(score_t,self_tm1,W):
			self_t = self.activation(score_t + T.dot(self_tm1,W))
			return self_t
		activation_probs,_ = theano.scan(
				step,
				sequences     = activation_score,
				outputs_info  = self.h0,
				non_sequences = self.W
			)
		return activation_probs

if __name__=='__main__':
	print Recurrent(3).mean(T.matrix('visible'))


