# model generation and save

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing as mp

en_fname = './data/prepro_en_wiki.txt'
model_fname = './model/taekeun/en.bin'

model = Word2Vec(LineSentence(en_fname), size=300, workers=mp.cpu_count(), sg=1)

model.save(model_fname)
