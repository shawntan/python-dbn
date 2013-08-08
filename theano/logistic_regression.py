import theano
import theano.tensor as T
import numpy         as np
class LogisticRegression:
	def __init__(self,features,classes,learning_rate=0.1):
		self.b = theano.shared(
				value = np.zeros((classes,),dtype=theano.config.floatX),
			)
		self.W = theano.shared(
				value = np.zeros((features,classes),dtype=theano.config.floatX),
			)

		self.x = T.matrix()
		self.y = T.ivector()

	def negative_log_likelihood(self,y):
		return -T.mean(
					T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]
				)

	def train(self,training_data,batch_size):

		self.p_y_given_x = T.nnet.softmax(T.dot(self.x,self.W) + self.b)
		self.learning_rate = learning_rate
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.predict = theano.function(
				inputs  = [self.x],
				outputs = self.y_pred)


		data_x,data_y = training_data
		train_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=True)
		train_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=True)
		train_y = T.cast(train_y,'int32')
		index = T.lscalar() 
		
		x,y = self.x, self.y
		cost = self.negative_log_likelihood(y)
		g_W  = T.grad(cost=cost,wrt=self.W)
		g_b  = T.grad(cost=cost,wrt=self.b)
		updates = [ ( self.W, self.W - self.learning_rate * g_W ),
					( self.b, self.b - self.learning_rate * g_b ) ]

		train_model = theano.function(
				inputs  = [index],
				outputs = cost,
				updates = updates,
				givens  = {
					x: train_x[index * batch_size:(index + 1) * batch_size],
					y: train_y[index * batch_size:(index + 1) * batch_size],
				}
			)
		print "Compiled functions."
		for _ in range(100000):
			print train_model(0)

if __name__ == '__main__':
	classifier = LogisticRegression(2,2)
	data_x = np.array(
			[[0,0],
			 [0,1],
			 [1,0],
			 [1,1]]
		)
	data_y = np.array([0,1,1,0])
	classifier.train((data_x,data_y),3)
	print classifier.predict(data_x)
			
