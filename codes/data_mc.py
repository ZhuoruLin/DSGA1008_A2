import os
import torch
import collections

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.wordcount = collections.Counter()

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path,voc_size=10000):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'),voc_size)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'),voc_size)
        self.test = self.tokenize(os.path.join(path, 'test.txt'),voc_size)

    def tokenize(self, path,voc_size):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # UNKNOWN TOKEN
        UNKNOWN = '<UNK>'
        self.dictionary.add_word(UNKNOWN)
        #Count words
        with open(path, 'r') as f:
        	for line in f:
        		words = line.split() + ['<eos>']
        		self.dictionary.wordcount.update(words)
        #Get Most common words
        most_common_counts = self.dictionary.wordcount.most_common(n=voc_size)
        most_common_words = [count[0] for count in most_common_counts]
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                	if word in most_common_words:
                		self.dictionary.add_word(word)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                	if word in most_common_words:
	                    ids[token] = self.dictionary.word2idx[word]
	                else:
	                	ids[token] = self.dictionary.word2idx[UNKNOWN]
	                token+=1


        return ids
