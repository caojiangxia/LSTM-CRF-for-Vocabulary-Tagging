import json
import numpy as np
import random
from random import choice
from tqdm import tqdm


def seq_padding(X):# 使用0在后面进行补充
    ML=len(X[0])
    return [x + [0] * (ML - len(x)) for x in X]


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

class DataLoader(object):
    def __init__(self, data, tag2id, char2id, batch_size=64, evaluation=False):
        self.batch_size = batch_size
        self.tag2id = tag2id
        self.char2id = char2id
        data = self.preprocess(data)
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        data += data[:2*batch_size]
        self.num_examples = len(data)
        self.data = [data[i:i+batch_size] for i in range(0, len(data)-batch_size, batch_size)]


    def preprocess(self, data):
        processed = []
        for d in data:
            text=[]
            ans=[]
            for word in d[0]:
                if self.char2id.get(word,0) != 0:
                    text.append(self.char2id[word])
                else :
                    text.append(1)
            for tag in d[1]:
                ans.append(self.tag2id[tag])
            processed += [(text,ans)]
        return processed

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):# 判断是不是相应的类型
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = list(zip(*batch))
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        text = np.array(seq_padding(batch[0]))
        tag= np.array(seq_padding(batch[1]))
        return (text,tag, orig_idx)