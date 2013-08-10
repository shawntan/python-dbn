import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
from rbm import RBM
class DBN(object):
	def __init__(self,layers,
			validation = 0.1,batch_size = 10):
		self.layers = layers
		self.rbms = [
				RBM(layers[i],layers[i+1])
				for i in xrange(len(layers)-1)
			]
		self.validation = validation
		self.batch_size = batch_size
	
	def fit(self,X,Y):
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
		
		print "Setup training functions..."
		params = []
		i_train_x, i_valid_x = train_x,valid_x
		for r in self.rbms:
			training_params = r.prepare_functions(n_train_batches,i_train_x,i_valid_x)
			params.append(training_params)
			i_train_x = r.t_transform(i_train_x)
			i_valid_x = r.t_transform(i_valid_x)
		for r,args in zip(self.rbms,params):
			r.train(*args)


