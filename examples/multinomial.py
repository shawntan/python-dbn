import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30)) 




def sample_k_times(prob,d):
	sample = theano_rng.multinomial(n=d,pvals=prob)
	return sample

counts = T.matrix('counts')
D = T.sum(counts,axis=1)
probs = counts / D
#D = D.reshape((1,)+D.shape)
samples, updates = theano.scan(
		fn=sample_k_times,
		sequences=[probs,D]
	)
print updates
smpl = theano.function([counts],samples,updates=updates)
for _ in xrange(10):
	print smpl(
			np.array([
				[5,3,2],
				[7,1,2],
				[3,3,4]
			],dtype=np.float32))
