import io
import sys
import numpy as np


class WordEmbeddingEngine:

    def __init__(self, fname, maxwords=-1):
        self.vdata = {}
        self.load_vectors(fname=fname,maxwords=maxwords)

    def load_vectors(self, fname, maxwords=-1):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n,d = map(int, fin.readline().split())
        self._word_count = n
        self._vector_dimension = d
        words_to_load = maxwords if maxwords > 0 else self._word_count
        len = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.vdata[tokens[0]] = map(float, tokens[1:])
            len = len+1
            if len >= words_to_load:
                break
        self._loaded_words = len
        print(f'{self._loaded_words} words with {self._vector_dimension} dimension loaded ')
        return self._loaded_words

    def distance(self, w1,w2):
        try :
            v1 = self._w2v(w1)
            v2 = self._w2v(w2)

        except Exception as ex:
            print(ex)
            return np.nan, np.nan

        norm = np.linalg.norm(v2 - v1)
        cosine = np.inner(v2,v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return norm , cosine

    def _w2v(self, word):
        if word not in self.vdata:
            raise Exception(f'{word} not exists in vdata')
        return np.array(list(self.vdata[word]))


if __name__ == '__main__':
    if len(sys.argv) < 5 :
        print (f'usage: python {sys.argv[0]} <vector_file_name> <max-words> <word1> <word2>')
        exit(1)
    fname = sys.argv[1]
    maxwords = int(sys.argv[2])
    w1 = sys.argv[3]
    w2 = sys.argv[4]

    print(f'loading words from {fname}')
    engine = WordEmbeddingEngine(fname=fname, maxwords=maxwords)
    norm , cosine = engine.distance(w1, w2)
    if norm >= 0:
        print(f'{w1[::-1]} - {w2[::-1]} : norm={norm} cosine={cosine}')

