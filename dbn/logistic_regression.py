import theano,math
import theano.tensor as T
import numpy         as np
import utils         as U
from base import BaseLayerPair

class LogisticRegression(BaseLayerPair):

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
