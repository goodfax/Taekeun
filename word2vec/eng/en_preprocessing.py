# nltk로 형태소 분석하여 전처리
import nltk

out_f = './data/prepro_en_wiki.txt'
in_f = './data/en_wiki.txt'

# encoding 관련 에러 뜰까봐 encoding='utf-8' 사용
output = open(out_f, 'wt', encoding='utf-8')

with open(in_f, 'r', encoding='utf-8') as rf:
    lines = rf.readlines()
    i = 0

    for line in lines:
        token = nltk.tokenize.word_tokenize(line)
        line = bytes(' '.join(token), 'utf-8').decode('utf-8') + '\n'
        output.write(line)

        i = i + 1
        if i % 10000 == 0:
            print('Preprocessed ' + str(i) + ' articles')
    output.close()
