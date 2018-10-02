import io
import sys
import numpy as np

vdata = {}

def load_vectors(fname, maxwords=-1):
    print (f'loading from {fname} ')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    words = maxwords if maxwords > 0 else n
    print(f'loading {words} words from {n} words with {d} vectors size ')
    len = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        vdata[tokens[0]] = map(float, tokens[1:])
        len = len+1
        if len >= words:
            break
    print(f'{len} words loaded')


def w2v(word):
    if word not in vdata:
        raise Exception(f'{word} not exists in vdata')
    return np.array(list(vdata[word]))

def distance(w1,w2):
    try :
        v1 = w2v(w1)
        v2 = w2v(w2)
    except Exception as ex:
        print(ex)
        return (-1,-1)
    norm = np.linalg.norm(v2 - v1)
    cosine = np.inner(v2,v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return norm , cosine

if __name__ == '__main__':
    if len(sys.argv) < 5 :
        print (f'usage: python {sys.argv[0]} <vector_file_name> <max-words> <word1> <word2>')
        exit(1)
    fname = sys.argv[1]
    maxwords = -1
    if len(sys.argv) > 2:
        maxwords = int(sys.argv[2])

    load_vectors(fname=sys.argv[1],maxwords=maxwords)

    if len(sys.argv) > 4:
        w1 = sys.argv[3]
        w2 = sys.argv[4]
    norm , cosine = distance(w1, w2)
    if norm >= 0:
        print(f'{w1[::-1]} - {w2[::-1]} : norm={norm} cosine={cosine}')

