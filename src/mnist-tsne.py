import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.datasets import mnist
from ggplot import *

# main
#============================================================
if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # print the shape before we reshape and normalize
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)

     # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255
    X = X_train
    y = y_train

    print (X.shape, y.shape)

    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

    df = pd.DataFrame(X,columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))

    X, y = None, None

    print (f'Size of the dataframe: {df.shape}')

    # use pca to reduce dimension from 748 to 50
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
    print (f'Cumulative explained variation for 50 principal components: {np.sum(pca_50.explained_variance_ratio_)}')

    # plot(df,rndperm)
    rndperm = np.random.permutation(df.shape[0])
    n_sne = 10000
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])

    print(f't-SNE done! Time elapsed: {time.time()-time_start} seconds')

    df_tsne = None
    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne-pca'] = tsne_pca_results[:,0]
    df_tsne['y-tsne-pca'] = tsne_pca_results[:,1]

    chart = ggplot( df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
                            + geom_point(size=70,alpha=0.1) \
                            + ggtitle("tSNE dimensions colored by Digit (PCA)")
    chart.show()
