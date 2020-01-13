# binary google word2vec 모델 불러오기
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('./model/google/GoogleNews-vectors-negative300.bin', binary=True)

# 해당단어와 유사한 단어출력
sbw = model.similar_by_word('computer')
print(sbw)

# 두 단어의 유사도 출력
sw = model.similarity('king', 'queen')
print(sw)
