# binary kyubyong word2vec 모델 불러오기
# https://github.com/Kyubyong/wordvectors
import gensim
from pprint import pprint

model = gensim.models.Word2Vec.load('./model/taekeun/ko.bin')

# 해당단어와 유사한 단어출력
sbw = model.similar_by_word('아이언맨')
pprint(sbw)

# 두 단어의 유사도 출력1
sw1 = model.similarity('한국', '미국')
print(sw1)

# 두 단어의 유사도 출력2
sw2 = model.similarity('엄마', '아빠')
print(sw2)


