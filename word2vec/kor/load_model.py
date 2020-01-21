# binary kyubyong word2vec 모델 불러오기
# https://github.com/Kyubyong/wordvectors
import gensim

# model = gensim.models.Word2Vec.load('./model/kyubyong/ko.bin')
model = gensim.models.Word2Vec.load('./model/taekeun/ko.bin')

# 해당단어와 유사한 단어출력
sbw = model.similar_by_word('컴퓨터')
print(sbw)

# 두 단어의 유사도 출력
sw = model.similarity('아빠', '엄마')
print(sw)