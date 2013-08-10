import sys,re,random
import numpy as np
from rbm import RBM
import theano.tensor as T
import theano
from logistic_regression import LogisticRegression
from layer import *
if __name__ == '__main__':
	data = np.vstack(100*(np.eye(8),))
	#np.random.shuffle(data)
	print data
	r = RBM(OneHotSoftmax(8),Sigmoid(3),lambda_2=0)
	r.fit(data)

	data = theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow=True)
	lr = 0.1
	x = T.matrix('inp')
	y = T.ivector('out')
	b = r.v_bias
	W = r.W.T
	tunables = [b] +  r.tunables[:-1]

	p_y_given_x = T.nnet.softmax(T.dot(r.t_transform(x),W) + b)
	cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
	
	gparams = T.grad(cost,tunables)

	updates = [ (param, param - lr * gparam) for param,gparam in zip(tunables,gparams) ]
	train_model = theano.function(
			inputs  = [x,y],
			outputs = cost,
			updates = updates,
		)
	
	data  = np.eye(8)
	label = np.asarray(np.arange(8),dtype=np.int32)
	for _ in xrange(100000): print train_model(data,label)
	print np.around(r.transform(data))
	identity = theano.function([x],p_y_given_x)
	print identity(data)
