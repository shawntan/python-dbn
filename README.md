Deep Belief Nets for Python
===========================
`python-dbn` aims to make experimenting with different deep learning 
architectures. Right now, it's capability is limited to specifying deep 
feed-forward networks. 

An example:
```python
from dbn        import DBN
from dbn.layers import *

net = DBN([
		OneHotSoftmax(8),
		Sigmoid(3)
	  ],3)

net.fit(train_data,train_labels)
net.predict(test_data)
```

###There's a still a lot left to do
- **Multiple visible layer sets per RBM**
  Multiple types of input for each RBM. Hopefully this will make
  experimenting different ways for data input to the network possible.
- **Auto-encoders**
  Building on to the DBN class to create an intuitive way to
  build auto-encoders just by specifying the layer dimensions.

###Getting the damn theory right
I'm still kinda new at this, so I'd appreciate any help I can get with the 
theory side of things.

Things I'm still unsure of that are already implemented:
- When do I stop training for RBMs? I've looked at Hinton's tutorial on
  training RBMs, but it doesn't seem to be helping for some of the tasks
  I've tested it with.
- With a softmax layer one-hot input, should my reconstruction be also be 
  turning just one neuron on based in the probability densities? Or should I not 
sample and leave the values as probabilities

