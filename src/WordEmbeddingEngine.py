import io
import sys
from collections import namedtuple
import sortedcontainers as sc

import numpy as np

MatchDistance = namedtuple('MatchDistance', ['source_word', 'target_word', 'norm','cosine'])

# key element for sort
def sortkey(elem):
    return elem.norm


class WordEmbeddingEngine:

    def __init__(self, fname, maxwords=-1):
        self.fname = fname
        self.word_count = 0
        self.loaded_words = 0
        self.vector_dimension = np.nan
        self._vdata = {}
        self._load_vectors(fname=self.fname,maxwords=maxwords)


    def distance(self, w1,w2):
        v1 = self[w1]
        v2 = self[w2]
        return MatchDistance(source_word=w1,
                        target_word=w2,
                        norm=np.linalg.norm(v2 - v1),
                        cosine=np.inner(v2,v1) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


    def top_match(self, source_word):
        topten = sc.SortedListWithKey(key=sortkey)
        for word in [w for w in self._vdata if w != source_word]:
            topten.add(self.distance(source_word, word))
            if len(topten) > 10:
                topten.pop()
        return topten

    def __getitem__(self, word):
        if word not in self._vdata:
            raise Exception(f'{word} not exists in {self.fname}')
        return self._vdata[word]


    def _load_vectors(self, fname, maxwords=-1):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n,d = map(int, fin.readline().split())
        self.word_count = n
        self.vector_dimension = d
        words_to_load = maxwords if maxwords > 0 else self.word_count
        len = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            self._vdata[tokens[0]] = np.array(list(map(float, tokens[1:])))
            len = len+1
            if len >= words_to_load:
                break
        self.loaded_words = len



if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print (f'usage: python {sys.argv[0]} <vector_file_name> <max-words> <word1> <word2>')
        exit(1)
    fname = sys.argv[1]
    maxwords = int(sys.argv[2])
    print(f'loading words from {fname}')
    engine = WordEmbeddingEngine(fname=fname, maxwords=maxwords)
    print(f'{engine.loaded_words} words with {engine.vector_dimension} dimension loaded from {engine.word_count} '
          f'words in {engine.fname}')

    if len(sys.argv) == 3:
        print(f'no words in arguments ',
               f'usage: python {sys.argv[0]} <vector_file_name> <max-words> <word1> <word2>')
        exit(0)

    w1 = sys.argv[3]

    # if one word supplied find top ten matches by distance
    if len(sys.argv) == 4:
        top_matches = engine.top_match(w1)
        log = '\n'.join([str(match) for match in top_matches])
        print(f'top match for {w1}: \n {log}')
        exit(0)

    # if input have 2 words find the distance
    w2 = sys.argv[4]

    try:
        match_dist = engine.distance(w1, w2)
        if match_dist.norm >= 0:
            print(f'{match_dist.source_word} - {match_dist.target_word} : norm={match_dist.norm} cosine={match_dist.cosine}')

    except Exception as ex:
        print(ex)