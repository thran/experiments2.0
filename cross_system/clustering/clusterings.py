from skll.metrics import kappa as kappa_metric

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from algorithms.spectralclustering import SpectralClusterer
from sklearn.manifold import TSNE, MDS, Isomap
import matplotlib.pylab as plt
colors = "rgbyk"
markers = "o^s*+x"


def remove_nans(df):
    filter = np.isnan(df).sum() < len(df) / 2
    df = df.loc[filter, filter]
    while np.isnan(df).sum().sum() > 0:
        worst = np.isnan(df).sum().argmax()
        df = df.loc[df.index != worst, df.index != worst]
    return df


def vectorization_pearson(answers):
    pivot = answers.pivot('student', 'item', 'correct')
    return remove_nans(pivot.corr(min_periods=10))


def vectorization_double_pearson(answers):
    pivot = answers.pivot('student', 'item', 'correct')
    return remove_nans(remove_nans(pivot.corr(min_periods=10)).corr())


def kappa(answers, min_periods=10):
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
    # correl[correl == 0] = np.nan

    return remove_nans(pd.DataFrame(correl, index=df.columns, columns=df.columns))


def spectral_clustering(X, clusters=2):
    X[X < 0] = 0
    sc = SpectralClusterer(X, kcut=X.shape[0] / 2, mutual=True)

    labels = sc.run(cluster_number=clusters, KMiter=50, sc_type=2)
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


def plot_clustering(items, xs, ys, labels=None, texts=None, shapes=None):
    labels = [0] * len(xs) if labels is None else labels
    texts = [""] * len(xs) if texts is None else texts
    shapes = [0] * len(xs) if shapes is None else shapes
    for item, x, y, label, text, shape in zip(items, xs, ys, labels, texts, shapes):
        plt.plot(x, y, markers[shape], color=colors[label])
        plt.text(x, y, text)
