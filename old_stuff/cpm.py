from rbm import RBM, sigmoid
import numpy as np
from scipy.sparse import coo_matrix

def softmax(x):
	maxes = np.amax(x,axis=1)
	maxes = maxes.reshape(maxes.shape[0],1)
	ex = np.exp(x-maxes)
	return ex/np.sum(ex,axis=1).reshape(maxes.shape[0], 1)

def visible_poisson(x,N):
	return N * softmax(x)

class CPM(RBM):
	def __init__(self, visible, hidden, epsilon = 0.1,
			act_fun_hidden = sigmoid,
			act_fun_visible = visible_poisson ):
		super(CPM,self).__init__(visible,hidden,epsilon,act_fun_hidden,act_fun_visible)
	
	def train(self, data, max_epochs = 100):
		num_examples = data.shape[0]
		N = np.sum(data, axis=1)
		N = N.reshape(N.shape[0],1)
		data = np.insert(data,0,1,axis=1)

		for epoch in range(max_epochs):
			# visible to hidden
			pos_hidden_activations = np.dot(data,self.weights)
			pos_hidden_probs       = self.act_fun_hidden(pos_hidden_activations)
			pos_hidden_states      = pos_hidden_probs > np.random.rand(num_examples,self.hidden + 1) # sample
			pos_associations       = np.dot(data.T,pos_hidden_probs)
			# hidden to visible
			neg_visible_activations = np.dot(pos_hidden_states,self.weights.T)
			neg_visible_expct       = self.act_fun_visible(neg_visible_activations,N)
			print neg_visible_expct
			neg_visible_expct[:,0]  = 1

			# visible to hidden
			neg_hidden_activations  = np.dot(neg_visible_expct,self.weights)
			neg_hidden_probs        = self.act_fun_hidden(neg_hidden_activations)
			neg_associations        = np.dot(neg_visible_expct.T, neg_hidden_probs)

			self.weights += self.epsilon * ((pos_associations - neg_associations)/num_examples)

			error = np.sum((data - neg_visible_expct) ** 2)
			print np.sum(neg_visible_expct,axis=1)
			print "Epoch %d: error is %0.5f"%(epoch,error)

if __name__ == "__main__":
	v = np.array([
		[  1.,  2.,  3. ],
		[  4., 10.,  6. ],
		[  1.,  2.,  4. ]
	], dtype=np.float64)
	print v
	net = CPM(3,5)
	net.train(v,1000)
