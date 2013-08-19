import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
from base import BaseLayerPair


class RBM(BaseLayerPair):
	def __init__(self, visible, hidden, **kwargs):
		kwargs['lambda_2'] = 0.0
		self.v = visible
		self.h = hidden
		inputs = self.v.size
		outputs = self.h.size
		
		super(RBM,self).__init__(inputs,outputs,**kwargs)
		self.h_bias       = self.bias
		self.h_bias_delta = self.bias_delta

		self.v_bias       = U.create_shared(np.zeros(self.v.size))
		self.v_bias_delta = U.create_shared(np.zeros(self.v.size))

		self.tunables += [self.v_bias]
		self.deltas   += [self.v_bias_delta]


	def t_transform(self,v):
		return self.h.activation(T.dot(v,self.W) + self.h_bias)

	def gibbs_hvh(self,h_sample):
		v_activation_score = T.dot(h_sample,self.W.T) + self.v_bias
		v_activation_probs, v_sample, v_updates = self.v.sample(v_activation_score)
		h_activation_score = T.dot(v_sample,self.W)   + self.h_bias
		h_activation_probs, h_sample, h_updates  = self.h.sample(h_activation_score)
		return v_activation_score,v_activation_probs,v_sample,\
			   h_activation_score,h_activation_probs,h_sample

	def gibbs_vhv(self,v_sample):
		h_activation_score = T.dot(v_sample,self.W)   + self.h_bias
		h_activation_probs, h_sample, h_updates = self.h.sample(h_activation_score)
		v_activation_score = T.dot(h_sample,self.W.T) + self.v_bias
		v_activation_probs, v_sample, v_updates  = self.v.sample(v_activation_score)
		return h_activation_score,h_activation_probs,h_sample,\
			   v_activation_score,v_activation_probs,v_sample

	def free_energy(self, v_sample):
		"""
		this is only for binary units!
		"""
		h_activation_score = T.dot(v_sample,self.W) + self.h_bias
		v_bias_term = T.dot(v_sample,self.v_bias)
		hidden_term = T.sum(T.log(1 + T.exp(h_activation_score)), axis=1)
		return - hidden_term - v_bias_term

	def reconstruction_cost(self,updates,nv_activation_scores,data):
		"""
		Cross entropy
		"""
		return T.mean(T.sum(
					data * T.log(self.v.activation(nv_activation_scores)) +
						(1 - data) * T.log(1 - self.v.activation(nv_activation_scores)),
				axis = 1))

	def regularisation(self):
		return self.lambda_2 * sum( T.sum(p**2) for p in self.tunables )

	def cost_updates(self,lr,data,k=1):
		ph_activation_scores = T.dot(data,self.W) + self.h_bias
		ph_activation_probs, ph_samples, ph_updates  = self.h.sample(ph_activation_scores)

		chain_start = ph_samples

		[nv_activation_scores,nv_activation_probs,nv_samples,\
		 nh_activation_scores,nh_activation_probs,nh_samples], updates = \
		theano.scan(
				 self.gibbs_hvh,
				 outputs_info = [None,None,None,None,None,chain_start],
				 n_steps      = k
			)
		chain_end = nv_samples[-1]
		cost = T.mean(self.free_energy(data))\
				- T.mean(self.free_energy(chain_end))\
				 + self.regularisation()

		gparams = T.grad(cost,self.tunables,consider_constant=[chain_end])

		alpha = T.cast(self.momentum,dtype=theano.config.floatX)
		updates = [
				( param, param - ( alpha * prev_chg + gparam * lr ) )
		   		for gparam,param,prev_chg in zip(gparams,self.tunables,self.deltas)
		   ] + [
				( prev_chg, alpha * prev_chg + gparam * lr )
				for prev_chg,gparam in zip(self.deltas,gparams)
		   ]# + ph_updates + nv_updates + nh_updates

		monitoring_cost = self.reconstruction_cost(updates,nv_activation_scores[-1],data)

		return monitoring_cost,updates

	def prepare_functions(self,n_train_batches,train_x,valid_x,train_y=None,valid_y=None):
		index   = T.lscalar('index')
	
		print "Compiling training function..."
		x,rep,val,lr    = T.matrix('x'), T.matrix('rep'), T.matrix('val'), T.scalar('lr')
		cost,updates    = self.cost_updates(lr,x)
		train_model = theano.function(
				inputs  = [index,lr],
				outputs = cost,
				updates = updates,
				givens  = {	x: train_x[index*self.batch_size:(index+1)*self.batch_size] },
			)

		compare_free_energy = theano.function(
				inputs  = [],
				outputs = T.abs_(T.mean(self.free_energy(val)) - T.mean(self.free_energy(rep))),
				givens  = { rep: train_x, val: valid_x }
			)
		print "Done."
		return n_train_batches,train_model,compare_free_energy



if __name__ == "__main__":
	from layers import *
	r = RBM(ReplicatedSoftmax(6),Sigmoid(3))
	training_data = np.array([	
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0],
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0],
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0],
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0],
	])

	r.fit(training_data)
