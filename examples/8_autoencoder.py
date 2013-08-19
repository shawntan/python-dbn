import sys,re,random
import numpy as np
from dbn        import DBN 
from dbn.layers import *
import theano.tensor as T
import theano
if __name__ == '__main__':
	data = np.hstack((np.eye(8),np.arange(8).reshape((8,1))))
	data = np.vstack(100*(data,))
	np.random.shuffle(data)
	
	net = DBN([
				OneHotSoftmax(8),
				Sigmoid(3)
			],8,max_epochs=1000)
	net.fit(data[:,:-1],data[:,-1])
	print net.predict(np.eye(8,dtype=np.float32))

