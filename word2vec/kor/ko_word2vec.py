# model generation and save

from gensim.models import Word2Vec

ko_fname = './data/prepro_ko_wiki.txt'
model_fname = './model/taekeun/ko.bin'

corpus = [sent.strip().split(" ") for sent in open(ko_fname, 'r', encoding='utf-8').readlines()]
model = Word2Vec(corpus, size=200, workers=4, sg=1)
model.save(model_fname)

print('Finish')
