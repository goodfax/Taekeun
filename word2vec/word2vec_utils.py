# word2vec에 필요한 utils

from tqdm import tqdm


class SentenceIterator:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in tqdm(open(self.filepath, 'r', encoding='utf-8'), total=int(4763709)):
            yield line.split()
