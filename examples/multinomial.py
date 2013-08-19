import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30)) 




def sample_k_times(prob):
	sample = theano_rng.multinomial(n=1,pvals=prob)
	return sample

probs = T.matrix('probs')
# Generate the components of the polynomial
samples, updates = theano.scan(
		fn=sample_k_times,
		sequences=probs
	)
# Sum them up
# Compile a function
smpl = theano.function([probs],samples)
for _ in xrange(10):
	print smpl(np.array([[0.5,0.5],[0.8,0.2]], dtype=np.float32))
