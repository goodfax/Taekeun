# word2vec에 필요한 utils

from tqdm import tqdm


class SentenceIterator:
    def __init__(self, filepath):
        self.filepath = filepath

    # int(4763709)는 prepro_en_wiki.txt의 라인 수
    # iterator에 크기 계산을 위한 소스를 넣으면 RAM의 용량을 초과하여
    # 미리 계산해 직접 값을 입력함. => 개선 필요.
    def __iter__(self):
        for line in tqdm(open(self.filepath, 'r', encoding='utf-8'), total=int(4763709)):
            yield line.split()
