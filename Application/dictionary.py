from utility import *

SOS_token = 0
EOS_token = 1
class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", SOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in word_tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexes_from_sentence(self, sentence):
        return [self.word2index[word] for word in word_tokenize(sentence)]

if __name__ == '__main__':
    trainingset = load('trainingset.pkl')

    dictionary = Dictionary()
    for s,e in trainingset:
        dictionary.add_sentence(s)
        dictionary.add_sentence(e)

    store(dictionary, 'dictionary')