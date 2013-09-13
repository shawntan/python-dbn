import utils
import theano
import theano.tensor as T
import numpy         as np
import utils         as U

def to_sparse_array(M):
	index  = np.array([ len(row) for row in M ],dtype=np.int32)
	values = np.array([ list(v)  for row in M for v in row ],dtype=np.int32)
	return index,values
def to_dense_array(M):
	index,values = to_sparse_arrays(M)
	dense = np.zeros((len(M),np.max(values[:,0])+1))
	prev = 0
	row  = 0
	for i in index:
		for col,val in values[prev:prev+i]:
			dense[row,col] = val
		row  += 1
		prev += i
	return dense

def sparse_dot(l,prev,values,W):
	row_data = values[T.arange(prev,prev+l)]
	row_weights = W[row_data[:,0]]
	sum_weights = T.sum(row_weights*row_data[:,1].reshape((l,1)),axis=0)
	return sum_weights,prev+l



if __name__ == "__main__":
	M = [[(1,2),(5,3),(10,1)],
		 [(0,2),(3,1)],
		 [(2,2),(8,4)]]
	index  = T.ivector('index')
	values = T.imatrix('values')
	prev   = T.iscalar('prev')
	initial_weights = U.initial_weights(11,3)
	W = U.create_shared(initial_weights)

	[output,_],updates = theano.scan(
			sparse_dot,
			sequences     = index,
			outputs_info  = [None,prev],
			non_sequences = [values,W]
		)

	f = theano.function(
			inputs = [index,values,prev],
			outputs = output
		)

	ind,val = to_sparse_array(M)
	print ind,val
	print f(ind,val,0)
