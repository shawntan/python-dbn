import numpy         as np
import utils         as U
import theano.tensor as T
import math

class BaseLayerPair(object):
	def __init__(self,inputs,outputs,
				 lr = 0.1,       batch_size = 10,  max_epochs = 100000,
				 momentum = 0.5, validation = 0.1, lambda_2 = 0.001,
				 lr_min = 0.1):
		self.momentum   = momentum
		self.lr         = lr
		self.lr_min     = lr_min
		self.batch_size = batch_size
		self.validation = validation
		self.max_epochs = max_epochs 
		self.lambda_2   = lambda_2


		self.W       = U.create_shared(U.initial_weights(inputs,outputs))
		self.W_delta = U.create_shared(np.zeros((inputs,outputs)))

		self.bias       = U.create_shared(np.zeros(outputs))
		self.bias_delta = U.create_shared(np.zeros(outputs))

		self.tunables = [self.W,       self.bias]
		self.deltas   = [self.W_delta, self.bias_delta]

	def fit(self,X,Y=None):
		print "Splitting validation and training set..."
		training_count  = int(X.shape[0]*(1-self.validation))
		validate_count  = X.shape[0] - training_count
		n_train_batches = int(math.ceil(training_count/float(self.batch_size)))
		print "Setting up shared training memory..."
		train_x = U.create_shared(X[:training_count])
		valid_x = U.create_shared(X[training_count:])

		if Y != None:
			train_y = T.cast(U.create_shared(Y[:training_count]),'int32')
			valid_y = T.cast(U.create_shared(Y[training_count:]),'int32')
		else:
			train_y = valid_y = None

		print "Total examples:", X.shape[0]
		print "train examples:", training_count
		print "valid examples:", validate_count 
		print "batches:       ", n_train_batches
		print "batch size:    ", self.batch_size
	
		self.train(*self.prepare_functions(n_train_batches,train_x,valid_x,train_y,valid_y))

	def train(self,n_train_batches,train_model,validate_model):
		max_epochs   = self.max_epochs
		patience     = max_epochs/10 
		patience_inc = 2
		gap_thresh   = 0.999
		val_freq     = min(n_train_batches,patience/2)
		epoch        = 0

		lr_denom     = float(1/self.lr_min)
		lr_numer     = float(self.lr/self.lr_min)
		curr_lr      = lr_numer / lr_denom
		best_error   = np.inf
		best_params  = None
		iter_no = 0

		while (epoch < max_epochs) and (patience > iter_no):
			total_error = 0
			for batch_index in xrange(n_train_batches):
				error = train_model(batch_index,curr_lr)
				total_error += error
				iter_no = epoch * n_train_batches + batch_index
				
				if epoch >= 1 and (iter_no + 1) % val_freq == 0:
					val_error = validate_model()
					if val_error < best_error * gap_thresh:
						patience = max(patience, iter_no * patience_inc)
						if best_params:
							lr_numer = math.sqrt(lr_numer)
							curr_lr  = lr_numer / lr_denom
						best_error = val_error 
						print "Saving parameters..."
						best_params = [ p.get_value() for p in self.tunables ]
			
			avg_error = total_error/float(n_train_batches-1)

			print "Training error:  ", avg_error
			print "Validation error:", validate_model()
			print "Patience:        ", patience
			print "Iteration:       ", iter_no
			print
			epoch += 1
		
		for p,best_p in zip(self.tunables,best_params): p.set_value(best_p)

