# model generation and save

from word2vec.word2vec_utils import SentenceIterator
from gensim.models import Word2Vec

en_fname = './data/prepro_en_wiki.txt'
model_fname = './model/taekeun/en.bin'

corpus = SentenceIterator(en_fname)

print('Start')

# default epoch: 5
model = Word2Vec(corpus, size=300, workers=4, sg=1)
model.save(model_fname)

print('Finish')
