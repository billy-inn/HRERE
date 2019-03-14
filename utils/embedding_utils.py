import numpy as np
import gensim

class Embedding:
	def __init__(self, f, corpus, max_document_length):
		if ".txt" in f:
			model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=False)
		else:
			model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=True)

		wordSet = set(['"'])
		for sen in corpus:
			words = sen.split()
			for w in words:
				if w in model:
					wordSet.add(w)

		vocab_size = len(wordSet)
		print("%d unique tokens have been found!" % vocab_size)
		embedding_dim = model.syn0.shape[1]
		word2id = {"<PAD>":0}
		id2word = {0:"<PAD>"}
		word2id = {"<UNK>":1}
		id2word = {1:"<UNK>"}
		embedding = np.zeros((vocab_size+2, embedding_dim))

		np.random.seed(0)
		#embedding[0, :] = np.random.uniform(-1, 1, embedding_dim)
		embedding[1, :] = np.random.uniform(-1, 1, embedding_dim)
		for i, word in enumerate(wordSet):
			word2id[word] = i+2
			id2word[i+2] = word
			embedding[i+2, :] = model[word]

		self.vocab_size = vocab_size + 2
		self.embedding_dim = embedding_dim
		self.word2id = word2id
		self.id2word = id2word
		self.embedding = embedding
		self.max_document_length = max_document_length
		self.position_size = self.max_document_length * 2 + 1
	
	def _text_transform(self, s, maxlen):
		if not isinstance(s, str):
			s = ""
		words = s.split()
		vec = []
		for w in words:
			if w == "''":
				w = '"'
			if w in self.word2id:
				vec.append(self.word2id[w])
			else:
				vec.append(1)
		for i in range(len(words), maxlen):
			vec.append(0)
		return vec[:maxlen]

	def _len_transform(self, s, maxlen):
		if not isinstance(s, str):
			s = ""
		length = len(s.split())
		return min(length, maxlen)

	def text_transform(self, s):
		return self._text_transform(s, self.max_document_length)

	def len_transform(self, s):
		return self._len_transform(s, self.max_document_length)

	def position_transform(self, s):
		x1, y1, x2, y2 = s
		vec1 = []
		vec2 = []
		for i in range(self.max_document_length):
			if i < x1:
				vec1.append(i-x1)
			elif i > y1:
				vec1.append(i-y1)
			else:
				vec1.append(0)
			if i < x2:
				vec2.append(i-x2)
			elif i > y2:
				vec2.append(i-y2)
			else:
				vec2.append(0)
		vec1 = [np.clip(p+self.max_document_length, 0, self.position_size-1) for p in vec1]
		vec2 = [np.clip(p+self.max_document_length, 0, self.position_size-1) for p in vec2]
		return [vec1, vec2]
