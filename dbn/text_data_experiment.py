import sys
from rbm import RBM
from nltk.corpus   import stopwords,gazetteers,names
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer


stopwords = stopwords.words('english')
names = [ line.strip() for line in open(sys.argv[1],'r') ]
word_counter = CountVectorizer(
		tokenizer=wordpunct_tokenize,
		stop_words=stopwords,
		binary=True,
		ngram_range=(1,1),
	)

print word_counter.fit_transform(names)

