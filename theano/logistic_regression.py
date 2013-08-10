import theano,math
import theano.tensor as T
import numpy         as np
import utils         as U
class LogisticRegression:
	def __init__(self,features,classes,
				 lr = 0.1,       batch_size = 10,  max_epochs = 100000,
				 momentum = 0.5, validation = 0.1, lambda_2 = 0.001):
		self.momentum   = momentum
		self.lr         = lr
		self.batch_size = batch_size
		self.validation = validation
		self.max_epochs = max_epochs 
		self.lambda_2   = lambda_2

		self.W       = U.create_shared(U.initial_weights(features,classes))
		self.W_delta = U.create_shared(np.zeros((features,classes)))

		self.bias       = U.create_shared(np.zeros(features))
		self.bias_delta = U.create_shared(np.zeros(features))

		self.tunables = [self.W,       self.bias]
		self.deltas   = [self.W_delta, self.bias_delta]

	def activation_probability(self,x):
		return T.nnet.softmax(T.dot(x,self.W) + self.bias)
	
	def t_predict(self,x):
		return T.argmax(self.activation_probability(x), axis=1)


	def cost_updates(self,lr,x,y):
		alpha = T.cast(self.momentum,dtype=theano.config.floatX)
		cost    = -T.mean(T.log(self.activation_probability(x))[T.arange(y.shape[0]),y])
		gparams =  T.grad(cost,self.tunables)

		updates = [
				( param, param - ( alpha * prev_chg + gparam * lr ) )
		   		for param,gparam,prev_chg in zip(self.tunables,gparams,self.deltas)
		   ] + [
				( prev_chg, alpha * prev_chg + gparam * lr )
				for prev_chg,gparam in zip(self.deltas,gparams)
		   ]
		return cost,updates

	def fit(self,X,Y):
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
	def error(self,x,y):
		return T.mean(T.neq(self.t_predict(x), y))

	def prepare_functions(self,n_train_batches,train_x,valid_x,train_y,valid_y):
		index = T.lscalar()	
		x,y,lr = T.matrix(), T.ivector(), T.scalar()
		cost,updates = self.cost_updates(lr,x,y)
		error = self.error(x,y)
		train_model = theano.function(
				inputs  = [index,lr],
				outputs = error,
				updates = updates,
				givens  = {
					x: train_x[index * self.batch_size:(index + 1) * self.batch_size],
					y: train_y[index * self.batch_size:(index + 1) * self.batch_size],
				}
			)

		validate_model = theano.function(
				inputs  = [],
				outputs = error, 
				givens  = { x: valid_x, y: valid_y }
			)
		return n_train_batches,train_model,validate_model

	def train(self,n_train_batches,train_model,validate_model):
		max_epochs   = self.max_epochs
		patience     = max_epochs/10 
		patience_inc = 2
		gap_thresh   = 0.999
		val_freq     = min(n_train_batches,patience/2)
		epoch        = 0
		curr_lr      = float(self.lr)
		best_error   = np.inf
		best_params = None
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
							curr_lr = math.sqrt(curr_lr)
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

		data = T.matrix('inputs')
		self.predict = theano.function(inputs = [data],outputs = self.t_predict(data))
	

if __name__ == '__main__':
	classifier = LogisticRegression(2,2)
	data = np.array(
			[[0,0,0],
			 [0,1,1],
			 [1,0,1],
			 [1,1,0]]
		)

	data = np.vstack(100*(data,))
	np.random.shuffle(data)
	classifier.fit(data[:,[0,1]],data[:,2])
