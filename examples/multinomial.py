import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30)) 




def sample_k_times(prob,d):
	sample = theano_rng.multinomial(n=d[0],pvals=prob)
	return sample

probs = T.matrix('probs')
D     = T.matrix('D')
samples, updates = theano.scan(
		fn=sample_k_times,
		sequences=[probs,D]
	)
smpl = theano.function([probs,D],samples,updates=updates)
for _ in xrange(10):
	print smpl(
			np.array([
				[0.5,0.3,0.2],
				[0.7,0.1,0.2],
				[0.3,0.3,0.4]
			],dtype=np.float32),
			np.array([[3],[2],[1]],dtype=np.float32))
