import numpy         as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import math
class RBM(object):
	def __init__(self, visible, hidden, lr = 0.1,batch_size=400,training_epochs=100000,
				 act_fun_hidden  = T.nnet.sigmoid,
				 act_fun_visible = T.nnet.sigmoid):
		self.h_activation = act_fun_hidden
		self.v_activation = act_fun_visible
		self.n_visible    = visible
		self.n_hidden     = hidden
		self.lr           = lr
		self.batch_size   = batch_size
		self.training_epochs = training_epochs

		self.theano_rng = RandomStreams(
				np.random.RandomState(1234).randint(2**30)
			)
		self.W = theano.shared(
				value = np.asarray(np.random.uniform(
					low  = -4 * np.sqrt(6./(visible+hidden)),
					high =  4 * np.sqrt(6./(visible+hidden)),
					size = (visible,hidden)),
					dtype = theano.config.floatX
				),
				name = 'W'
			)
		self.h_bias = theano.shared(
				value = np.zeros(hidden,dtype=theano.config.floatX),
				name  = 'hbias'
			)
		self.v_bias = theano.shared(
				value = np.zeros(visible,dtype=theano.config.floatX),
				name  = 'vbias'
			)
		self.tunables = [self.W,self.h_bias,self.v_bias]
	
	def sample_h_given_v(self,v_sample):
		dot_product  = T.dot(v_sample,self.W) + self.h_bias
		h_activation = self.h_activation(dot_product)

		h_sample = self.theano_rng.binomial(
				size  = h_activation.shape,
				n     = 1,
				p     = h_activation,
				dtype = theano.config.floatX
			)
		return dot_product,h_activation,h_sample

	def sample_v_given_h(self,h_sample):
		dot_product  = T.dot(h_sample,self.W.T) + self.v_bias
		v_activation = self.v_activation(dot_product)

		v_sample = self.theano_rng.binomial(
				size  = v_activation.shape,
				n     = 1,
				p     = v_activation,
				dtype = theano.config.floatX
			)
		return dot_product,v_activation,v_sample

	def gibbs_hvh(self,h_sample):
		v_activation_score, v_activation_probs, v_sample = self.sample_v_given_h(h_sample)
		h_activation_score, h_activation_probs, h_sample = self.sample_h_given_v(v_sample)
		return v_activation_score,v_activation_probs,v_sample,\
			   h_activation_score,h_activation_probs,h_sample

	def gibbs_vhv(self,v_sample):
		h_activation_score, h_activation_probs, h_sample = self.sample_h_given_v(v_sample)
		v_activation_score, v_activation_probs, v_sample = self.sample_v_given_h(h_sample)
		return h_activation_score,h_activation_probs,h_sample,\
			   v_activation_score,v_activation_probs,v_sample
	
	def free_energy(self, v_sample):
		h_activation_score = T.dot(v_sample,self.W) + self.h_bias
		v_bias_term = T.dot(v_sample,self.v_bias)
		hidden_term = T.sum(T.log(1 + T.exp(h_activation_score)), axis=1)
		return - hidden_term - v_bias_term

	def reconstruction_cost(self,updates,nv_activation_scores,data):
		"""
		Cross entropy
		"""
		return T.mean(
				T.sum(
						data * T.log(self.v_activation(nv_activation_scores)) +
						(1 - data) * T.log(1 - self.v_activation(nv_activation_scores)),
					axis = 1
				)
			)
	def cost_updates(self,data,k=1):
		ph_activation_scores, ph_activation_probs, ph_samples = self.sample_h_given_v(data)
		chain_start = ph_samples

		[nv_activation_scores,nv_activation_probs,nv_samples,\
		 nh_activation_scores,nh_activation_probs,nh_samples], updates = \
		theano.scan(
				 self.gibbs_hvh,
				 outputs_info = [None,None,None,None,None,chain_start],
				 n_steps      = 1
			)
		chain_end = nv_samples[-1]
		cost = T.mean(self.free_energy(data)) - T.mean(self.free_energy(chain_end))
		gparams = T.grad(cost,self.tunables,consider_constant=[chain_end])
		updates = {
				param: param - gparam * T.cast(self.lr,dtype=theano.config.floatX)
		   		 for gparam,param in zip(gparams,self.tunables)
		   }
		monitoring_cost = self.reconstruction_cost(updates,nv_activation_scores[-1],data)

		return monitoring_cost,updates

	def fit(self,X):
		print "Setting up cost and update functions..."
		cost,updates    = self.cost_updates(X)
		n_train_batches = int(math.ceil(X.shape[0]/float(self.batch_size)))
		
		print "Setting up shared training memory..."
		train_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)
		index   = T.lscalar()
		x       = T.matrix('x')
		
		print "Compiling training function..."
		train_rbm = theano.function(
				[index],
				cost,
				updates = updates,
				givens  = {	x: train_x[index*self.batch_size:(index+1)*self.batch_size] }
			)
		
		print "Done."
		for epoch in xrange(self.training_epochs):
			for batch_index in xrange(n_train_batches):
				print "Error:",train_rbm(batch_index)


if __name__ == "__main__":
	r = RBM(visible=6,hidden=2,lr=0.1)

	training_data = np.array([	
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0]
	])

	r.fit(training_data)

