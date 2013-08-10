import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U


class RBM(object):
	"""
	TODO:
	- Adaptive learning
	- PCD
	- Sparsity
	- Regularisation
		- L1 & L2
	- Early-stopping
	"""
	def __init__(self, visible, hidden,
				 lr = 0.1,       batch_size = 10,  max_epochs = 100000,
				 momentum = 0.5, validation = 0.1, lambda_2 = 0.01):

		self.v          = visible 
		self.h          = hidden
		self.momentum   = momentum
		self.lr         = lr
		self.batch_size = batch_size
		self.validation = validation
		self.max_epochs = max_epochs 
		self.lambda_2   = lambda_2

		self.W       = U.create_shared(U.initial_weights(self.v.size,self.h.size))
		self.W_delta = U.create_shared(np.zeros((self.v.size,self.h.size)))

		self.v_bias       = U.create_shared(np.zeros(self.v.size))
		self.v_bias_delta = U.create_shared(np.zeros(self.v.size))

		self.h_bias       = U.create_shared(np.zeros(self.h.size))
		self.h_bias_delta = U.create_shared(np.zeros(self.h.size))

		self.tunables = [self.W,       self.h_bias,       self.v_bias]
		self.deltas   = [self.W_delta, self.h_bias_delta, self.v_bias_delta]

	def t_transform(self,v):
		return self.h.activation(T.dot(v,self.W) + self.h_bias)

	def gibbs_hvh(self,h_sample):
		v_activation_score, v_activation_probs, v_sample = \
				self.v.sample(self.W.T,self.v_bias,h_sample)
		h_activation_score, h_activation_probs, h_sample = \
				self.h.sample(self.W,self.h_bias,v_sample)
		return v_activation_score,v_activation_probs,v_sample,\
			   h_activation_score,h_activation_probs,h_sample

	def gibbs_vhv(self,v_sample):
		h_activation_score, h_activation_probs, h_sample = \
				self.h.sample(self.W,self.h_bias,v_sample)
		v_activation_score, v_activation_probs, v_sample = \
				self.v.sample(self.W.T,self.v_bias,h_sample)
		return v_activation_score,v_activation_probs,v_sample,\
			   h_activation_score,h_activation_probs,h_sample

	def free_energy(self, v_sample):
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

	def cost_updates(self,data,lr,k=1):
		ph_activation_scores, ph_activation_probs, ph_samples = \
				self.h.sample(self.W,self.h_bias,data)
		chain_start = ph_samples

		[nv_activation_scores,nv_activation_probs,nv_samples,\
		 nh_activation_scores,nh_activation_probs,nh_samples], updates = \
		theano.scan(
				 self.gibbs_hvh,
				 outputs_info = [None,None,None,None,None,chain_start],
				 n_steps      = 1
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
		   ]

		monitoring_cost = self.reconstruction_cost(updates,nv_activation_scores[-1],data)

		return monitoring_cost,updates

	def prepare_functions(self,train_x,valid_x):
		index   = T.lscalar('index')
	
		print "Compiling training function..."
		x,rep,val,lr    = T.matrix('x'), T.matrix('rep'), T.matrix('val'), T.dscalar('lr')
		cost,updates    = self.cost_updates(x,lr)
		train_rbm = theano.function(
				inputs  = [index,lr],
				outputs = cost,
				updates = updates,
				givens  = {	x: train_x[index*self.batch_size:(index+1)*self.batch_size] },
			)

		compare_free_energy = theano.function(
				inputs  = [],
				outputs = T.abs_(T.mean(self.free_energy(val)) - T.mean(self.free_energy(rep))),
				givens  = { rep: train_x[:validate_count], val: valid_x }
			)
		print "Done."
		return n_train_batches,train_rbm,compare_free_energy


	def train(self,n_train_batches,train_rbm,compare_free_energy):
		max_epochs   = self.max_epochs
		patience     = max_epochs/10 
		patience_inc = 2
		gap_thresh   = 0.999
		val_freq     = min(n_train_batches,patience/2)
		epoch        = 0
		curr_lr      = float(self.lr)
		best_energy_gap = np.inf
		best_params = None
		iter_no = 0

		while (epoch < max_epochs) and (patience > iter_no):
			total_error = 0
			for batch_index in xrange(n_train_batches):
				error = train_rbm(batch_index,curr_lr)
				total_error += error
				iter_no = epoch * n_train_batches + batch_index
				
				if epoch >= 1 and (iter_no + 1) % val_freq == 0:
					val_energy_gap = compare_free_energy()
					
					if val_energy_gap < best_energy_gap * gap_thresh:
						patience = max(patience, iter_no * patience_inc)
						if best_params:
							curr_lr = math.sqrt(curr_lr)
						best_energy_gap = val_energy_gap
						print "Saving parameters..."
						best_params = [ p.get_value() for p in self.tunables ]
						

			avg_error = total_error/float(n_train_batches-1)

			print "Cross-entropy Error:", avg_error
			print "Energy difference:  ", compare_free_energy()
			print "Patience:           ", patience
			print "Iteration:          ", iter_no
			print
			epoch += 1

		#for p,best_p in zip(self.tunables,best_params): p.set_value(best_p)
		data = T.matrix('inputs')
		self.transform = theano.function(inputs = [data],outputs = self.t_transform(data))
	
	def fit(self,X):
		print "Splitting validation and training set..."
		training_count  = int(X.shape[0]*(1-self.validation))
		validate_count  = X.shape[0] - training_count
		n_train_batches = int(math.ceil(training_count/float(self.batch_size)))
		print "Setting up shared training memory..."
		train_x = theano.shared(np.asarray(X[:training_count],dtype=theano.config.floatX),borrow=True)
		valid_x = theano.shared(np.asarray(X[training_count:],dtype=theano.config.floatX),borrow=True)
		print "Total examples:", X.shape[0]
		print "train examples:", training_count
		print "valid examples:", validate_count 
		print "batches:       ", n_train_batches
		print "batch size:    ", self.batch_size
	
		self.train(*self.prepare_functions(train_x,valid_x))




if __name__ == "__main__":
	from layer import *
	r = RBM(Sigmoid(6),Sigmoid(3))
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
