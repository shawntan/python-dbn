import numpy as np

def sigmoid(x):
	return 1 / ( 1 + np.exp(-x) )

class RBM(object):
	def __init__(self, visible, hidden, epsilon = 0.1, act_fun_hidden = sigmoid, act_fun_visible = sigmoid):
		self.epsilon = epsilon
		self.weights = 0.1 * np.random.randn(visible + 1,hidden + 1)
		self.weights[0,:] = 0
		self.weights[:,0] = 0

		self.visible = visible
		self.hidden  = hidden 
		self.act_fun_hidden  = act_fun_hidden
		self.act_fun_visible = act_fun_visible
	
	def run_visible(self, data):
		num_examples = data.shape[0]
		hidden_states = np.ones((num_examples, self.hidden + 1))
		data = np.insert(data, 0, 1, axis = 1)
		hidden_activations = np.dot(data, self.weights)
		hidden_probs = self.act_fun_hidden(hidden_activations)
		hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.hidden + 1)
		hidden_states = hidden_states[:,1:]
		return hidden_states
	
	def run_hidden(self, data):
		num_examples = data.shape[0]
		visible_states = np.ones((num_examples, self.visible+ 1))
		data = np.insert(data, 0, 1, axis = 1)
		visible_activations = np.dot(data,self.weights.T)
		visible_probs = self.act_fun_visible(visible_activations)
		visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.visible+ 1)
		visible_states = visible_states[:,1:]
		return visible_states

	def train(self, data, max_epochs = 100):
		num_examples = data.shape[0]
		data = np.insert(data,0,1,axis=1)
		for epoch in range(max_epochs):
			pos_hidden_activations = np.dot(data,self.weights)
			pos_hidden_probs       = self.act_fun_hidden(pos_hidden_activations)
			pos_hidden_states      = pos_hidden_probs > np.random.rand(num_examples,self.hidden + 1)
			pos_associations       = np.dot(data.T,pos_hidden_probs)

			neg_visible_activations = np.dot(pos_hidden_states,self.weights.T)
			neg_visible_probs       = self.act_fun_visible(neg_visible_activations)
			neg_visible_probs[:,0]  = 1
			neg_hidden_activations  = np.dot(neg_visible_probs,self.weights)
			neg_hidden_probs        = self.act_fun_hidden(neg_hidden_activations)
			neg_associations        = np.dot(neg_visible_probs.T, neg_hidden_probs)

			self.weights += self.epsilon * ((pos_associations - neg_associations)/num_examples)

			error = np.sum((data - neg_visible_probs) ** 2)
			print "Epoch %d: error is %0.5f"%(epoch,error)


if __name__ == "__main__":
	r = RBM(6,2,0.1)

	training_data = np.array([	
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,1],
		[0,0,1,1,1,0]
	])

	r.train(training_data,100000)
	print r.weights
	for _ in range(10):
		print r.run_visible(np.array([
			[1,1,1,0,0,0],
			[0,0,1,0,1,1]
		]))
