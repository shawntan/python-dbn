import sys,re,random
import numpy as np
from dbn import DBN
from layers import *
from nltk.corpus   import stopwords,gazetteers,names
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
stopwords = stopwords.words('english')

def preprocess(sentence):
	sentence = sentence.lower()
	sentence = sentence.split()
	sentence = [ w for w in sentence if len(w) >= 3 ]
	sentence = [ re.sub(r'[0-9]','#',w) for w in sentence ]
	return ' '.join(sentence)

if __name__ == '__main__':
	hidden_units = 200
	names = [ preprocess(line.strip()) for line in open(sys.argv[1],'r') ]
	random.shuffle(names)
	word_counter = CountVectorizer(
			tokenizer=wordpunct_tokenize,
			stop_words=stopwords,
			binary=True,
			dtype=np.byte
		)
	data = word_counter.fit_transform(names)
	words = word_counter.get_feature_names()
	data = data.toarray()
	print data.shape
	_,vocab = data.shape

	n = DBN([ 
			Sigmoid(data.shape[1]),
			Sigmoid(hidden_units),
			Sigmoid(hidden_units/2)
		])
	n.fit(data,None)
	"""
	visible = r.run_hidden(np.eye(hidden_units))
	out = open('assoc_words','w')
	for f in range(hidden_units):
		out.write(' '.join( words[i] for i in range(len(words)) if visible[f,i] ) )
		out.write('\n')
	"""
