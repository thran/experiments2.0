from skll.metrics import kappa as kappa_metric

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from algorithms.spectralclustering import SpectralClusterer
from sklearn.manifold import TSNE, MDS, Isomap
import matplotlib.pylab as plt
colors = "rgbyk"


def vectorization_pearson(answers):
    pivot = answers.pivot('student', 'item', 'correct')
    return pivot.corr()


def vectorization_double_pearson(answers):
    pivot = answers.pivot('student', 'item', 'correct')
    return pivot.corr().corr()


def kappa(answers, min_periods=1):
    df = answers.pivot('student', 'item', 'correct')
    mat = df.as_matrix().T
    K = len(df.columns)
    correl = np.empty((K, K), dtype=float)
    mask = np.isfinite(mat)
    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue

            valid = mask[i] & mask[j]
            if valid.sum() < min_periods:
                c = np.nan
            elif i == j:
                c = 1.
            elif not valid.all():
                c = kappa_metric(ac[valid], bc[valid])
            else:
                c = kappa_metric(ac, bc)
            correl[i, j] = c
            correl[j, i] = c

    return pd.DataFrame(correl, index=df.columns, columns=df.columns)


def spectral_clustering(X, clusters=2):
    sc = SpectralClusterer(X, kcut=X.shape[0] / 2, mutual=True)

    labels = sc.run(cluster_number=2, KMiter=50, sc_type=2)
    return (sc.eig_vect[:, 1], sc.eig_vect[:, 2]), labels


def tsne(X, clusters=2):
    model = TSNE(learning_rate=300, n_iter=100000, init='pca')
    result = model.fit_transform(X)

    return result.T, None


def pca(X, clusters=2):
    model = PCA(n_components=2)
    result = model.fit_transform(X)

    return result.T, None


def isomap(X, clusters=2):
    model = Isomap(n_neighbors=15)
    result = model.fit_transform(X)

    return result.T, None


def mds(X, clusters=2):
    model = MDS(max_iter=1000)
    result = model.fit_transform(X)

    return result.T, None


def plot_clustering(items, xs, ys, labels, texts):
    for item, x, y, label, text in zip(items, xs, ys, labels, texts):
        plt.plot(x, y, "o", color=colors[label])
        plt.text(x, y, text)
