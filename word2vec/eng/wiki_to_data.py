# source code by 한국어 임베딩(book)에서 수정

import multiprocessing as mp

from gensim.corpora import WikiCorpus, Dictionary

in_f = "./data/enwiki-20200101-pages-articles.xml.bz2"
out_f = "./data/en_wiki.txt"

# cp949 에러 해결을 위해 encoding='utf-8' 추가
output = open(out_f, 'w', encoding='utf-8')

"""
#######################################################################
        This probably means that you are on Windows and you have
        forgotten to use the proper idiom in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce a Windows executable.
#######################################################################        
# 윈도우에서 multiprocess를 사용할때 fork()관련해서 생기는 에러
# 위 에러 해결을 위해 아래와 같은 코드 삽입
import multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()
"""

if __name__ == '__main__':
    mp.freeze_support()
    wiki = WikiCorpus(in_f, dictionary=Dictionary())
    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if i % 10000 == 0:
            print('Processed ' + str(i) + ' articles')
    output.close()
    print('Processing complete!')

