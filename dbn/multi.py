from itertools import *
class MultiInput(Linear):
	def __init__(self,layers):
		self.layers = layers

	def activation_probability(self,Ws,biass,inputss):
		activations = [
				 l.activation_probability(W,b,i)
				 for l,W,b,i in izip(self.layers,Ws,biass,inputss)
			]


		return zip(*activations)

	def sample(self,W,bias,inputs):
		samples = [
				 l.sample(W,b,i)
				 for l,W,b,i in izip(self.layers,Ws,biass,inputss)
			]


		return zip(*samples)




		

