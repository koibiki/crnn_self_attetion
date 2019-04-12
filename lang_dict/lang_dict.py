from config import cfg


class LanguageDict:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):

        [self.vocab.add(c) for c in cfg.CHAR_VECTOR]

        self.vocab = sorted(self.vocab)

        self.word2idx[''] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word
