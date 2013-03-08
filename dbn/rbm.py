import numpy as np

class rbm:
	def __init__(self,visible,hidden,epsilon):
		self.epsilon = epsilon
		self.weights = np.random.randn(visible,hidden)
		self.visible_bias = np.random.randn(visible)
		self.hidden_bias  = np.random.randn(hidden)
		self.visible_state = np.random.randint(0,2,visible)
		self.hidden_state  = np.random.randint(0,2,hidden)

	def hidden_activation_probs(self):
		return sigmoid(
				np.dot(self.visible_state,self.weights) +
				self.hidden_bias
			)

	def visible_activation_probs(self):
		return sigmoid(
				np.dot(self.weights,self.hidden_state) +
				self.visible_bias
			)

	def sample_states(self,states,activations):
		states[:] = activations > np.random.rand(states.shape[0])
	
	def visible_hidden(self):
		act_probs = self.hidden_activation_probs()
		self.sample_states(self.hidden_state,act_probs)
	
	def hidden_visible(self):
		act_probs = self.visible_activation_probs()
		self.sample_states(self.visible_state,act_probs)
	
	def delta_weight(self):
		#print "Start:",self.visible_state
		self.visible_hidden()
		#print "Visible->Hidden"
		#print "Visible: ",self.visible_state
		#print "Hidden: ",self.hidden_state
		positive = np.matrix(self.visible_state).T * np.matrix(self.hidden_state)
		#print "Positive:"
		self.hidden_visible()
		#print positive
		#print "Visible->Hidden"
		#print "Visible: ",self.visible_state
		#print "Hidden: ",self.hidden_state
		negative = np.matrix(self.visible_state).T * np.matrix(self.hidden_state)
		#print "Negative:"
		#print negative
		delta = self.epsilon*(positive-negative)
		#print delta
		return delta



		

def sigmoid(x):
	return 1 / ( 1 + np.exp(-x) )

r = rbm(5,3,0.01)
print r.weights

r.visible_state[:] = [ 0, 0, 1, 0, 1 ] 
for _ in range(100):
	r.weights += r.delta_weight()
r.visible_state[:] = [ 0, 1, 0, 0, 1 ] 
for _ in range(100):
	r.weights += r.delta_weight()

"""
r = rbm(5,3)
print r.visible_state
shit = 5 * np.ones(5)
print shit
print shit * r.visible_state

for i in range(100):
	r.visible_hidden()
	r.hidden_visible()
	print r.visible_state
"""

