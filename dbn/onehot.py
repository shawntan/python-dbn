import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U

initial_weights = U.initial_weights(8,3)
W = U.create_shared(initial_weights)
data = T.imatrix('data')
label = T.matrix('label')
def construct(bits_set,W):
	return W[bits_set].sum(axis=0)

output,updates = theano.scan(
		construct,
		sequences = data,
		non_sequences = W
	)

cost = T.mean(0.5*T.sum((output - label)**2,axis=1))


grad = T.grad(cost,wrt=W)

x = np.arange(8,dtype=np.int32).reshape(8,1)
y = np.array(
		[[0,0,0],
		 [0,0,1],
		 [0,1,0],
		 [0,1,1],
		 [1,0,0],
		 [1,0,1],
		 [1,1,0],
		 [1,1,1]],
	dtype=np.float32)

f = theano.function(
		inputs  = [data,label],
		outputs = output,
		updates = [(W,W - 0.5*grad)]
	)

