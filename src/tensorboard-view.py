import sys
import os

# for presentations :
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from src.wordEmbeddingEngine import WordEmbeddingEngine

LOG_DIR = '../tensorboard'

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

        embedding.metadata_path = metadata

        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

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
        print(f'no words in arguments view all loaded words')
        tensorboard_view(engine._vdata)
        exit(0)

    w1 = sys.argv[3]

    # if word supplied find top ten matches by distance
    if len(sys.argv) == 4:
        top_matches = engine.top_match(w1)
        log = '\n'.join([engine.print_match(m) for m in top_matches])
        print(f'top match:\n{log}')
        words = [m.target_word for m in top_matches]
        words.append('blue')
        tensorboard_view(engine._vdata, words)
        exit(0)

# refernce
# ----------------------------------------------------------------------------

def tsne_view(self, words):
    tsne = TSNE()
    embedding = tf.Variable(self.m[:100])
    # embed_tsne = tsne.fit_transform(embedding[:words, :])
    embed_tsne = tsne.fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
