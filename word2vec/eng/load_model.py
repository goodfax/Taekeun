# taekeun's word2vec 모델 불러오기
import gensim

model = gensim.models.Word2Vec.load('./model/taekeun/en.bin')

# 해당단어와 유사한 단어출력
sbw = model.wv.similar_by_word('computer')
print(sbw)

# 두 단어의 유사도 출력
sw = model.wv.similarity('king', 'queen')
print(sw)

# vocabulary size
vocab_size = model.wv.vectors.shape
print(vocab_size)
