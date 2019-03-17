from gensim.scripts.glove2word2vec import glove2word2vec
import os

os.system("wget http://nlp.stanford.edu/data/glove.840B.300d.zip")
os.system("unzip glove.840B.300d.zip")
os.system("rm glove.840B.300d.zip")

glove_file = "glove.840B.300d.txt"
word2vec_file = "glove.840B.300d.w2v.txt"
_ = glove2word2vec(glove_file, word2vec_file)

os.system("mv %s %s" % (word2vec_file, glove_file))
