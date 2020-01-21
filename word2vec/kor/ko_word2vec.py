# model generation and save

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing as mp

ko_fname = './data/prepro_ko_wiki.txt'
model_fname = './model/taekeun/ko.bin'

# corpus = [sent.strip().split(" ") for sent in open(ko_fname, 'r', encoding='utf-8').readlines()]
model = Word2Vec(LineSentence(ko_fname), size=100, workers=mp.cpu_count()-1, sg=1)
model.save(model_fname)
