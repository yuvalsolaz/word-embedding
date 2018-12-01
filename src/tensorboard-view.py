import sys
import os

# for presentations :
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from wordEmbeddingEngine import WordEmbeddingEngine

LOG_DIR = './tensorboard'

def tensorboard_view(v_data, words=None):
    df = pd.DataFrame.from_records(data=v_data)
    tf_data = tf.Variable(df.values.transpose()) if words is None else tf.Variable(df[words].values.transpose())

    ## Running Tensorlow Session
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name

        # Link this tensor to its metadata(Labels) file
        metadata = os.path.join(LOG_DIR, 'metadata.tsv')
        with open(metadata, 'w+') as metadata_file:
            _words = v_data.keys() if words is None else words
            for word in _words:
                metadata_file.write(f'{word}\n')

        embedding.metadata_path = 'metadata.tsv'

        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        print(f'create logs on {LOG_DIR}')

if __name__ == '__main__':
    if len(sys.argv) < 3 :
        print (f'usage: python {sys.argv[0]} <vector_file_name> <max-words> <word>')
        exit(1)
    fname = sys.argv[1]
    maxwords = int(sys.argv[2])
    print(f'loading words from {fname}')
    engine = WordEmbeddingEngine(fname=fname, maxwords=maxwords)
    print(f'{engine.loaded_words} words with {engine.vector_dimension} dimension loaded from {engine.word_count} '
          f'words in {engine.fname}')

    if len(sys.argv) == 3:
        print(f'no words file in arguments view all loaded words')
        tensorboard_view(engine._vdata)
        exit(0)

    word_file = sys.argv[3]

    # if word file supplied load and view words from given file
    words = []
    try:
        with open(word_file) as file:
            for line in file:
                words.append(line.rstrip())
        print(f'view {len(words)}  words ')
        tensorboard_view(engine._vdata, words)
    except FileExistsError:
        print (f'error opening file {word_file}')


