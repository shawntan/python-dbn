import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
from logistic_regression import LogisticRegression
from rbm import RBM
class DBN(LogisticRegression):
	def __init__(self,layers,outputs,**kwargs):
		self.layers = layers
		self.units  = [ RBM(layers[i],layers[i+1],**kwargs)
							for i in xrange(len(layers)-1) ]
		for i,u in enumerate(self.units):
			u.W.name    = "weight_%d"%i
			u.bias.name = "bias_%d"%i
		super(DBN,self).__init__(layers[-1].size,outputs,**kwargs)

	def prepare_functions(self,n_train_batches,train_x,valid_x,train_y,valid_y):
		print "Setup training functions..."
		params   = []
		x,y,lr = T.matrix(), T.ivector(), T.scalar()
		index   = T.lscalar('index')
		i_train_x, i_valid_x, i_x = train_x,valid_x, x

		for r in self.units:
			training_params = r.prepare_functions(n_train_batches,i_train_x,i_valid_x)
			params.append(training_params)
			self.tunables.extend([r.W,r.bias])
			self.deltas.extend([r.W_delta,r.bias_delta])
			# transform lower layer expression into higher layer expression
			i_train_x = r.t_transform(i_train_x)
			i_valid_x = r.t_transform(i_valid_x)
			i_x       = r.t_transform(x)
		
		# i_train_x is now the highest layer before the LogRes layer
		# cost_updates from LogisticRegression will create gradients
		# for all weights added to self.tunables and self.deltas

		cost, updates = self.cost_updates(lr,i_x,y)
		error = self.error(i_x,y)
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
		self.predict = theano.function(
				inputs  = [x],
				outputs = self.t_predict(i_x)
			)
		return n_train_batches,train_model,validate_model,params
	
	def train(self,n_train_batches,train_model,validate_model,params):
		# Pre-training
		print "Pre-training phase..."
		for r,params in zip(self.units,params): r.train(*params)
		print "Done."
		print
		print "Back propagation..."
		super(DBN,self).train(n_train_batches,train_model,validate_model)
