from eunjeon import Mecab

out_f = './data/prepro_ko_wiki.txt'
in_f = './data/ko_wiki.txt'

me = Mecab()
# cp949 에러 해결을 위해 encoding='utf-8' 추가
output = open(out_f, 'wt', encoding='utf-8')

with open(in_f, 'r', encoding='utf-8') as rf:
    lines = rf.readlines()
    i = 0

    for line in lines:
        temp_arr = me.morphs(line)
        line = bytes(' '.join(temp_arr), 'utf-8').decode('utf-8') + '\n'
        output.write(line)

        i = i + 1
        if i % 10000 == 0:
            print('Preprocessed ' + str(i) + ' articles')
    output.close()
    print('Preprocessing complete!')
