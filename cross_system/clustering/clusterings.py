from skll.metrics import kappa

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from algorithms.spectralclustering import SpectralClusterer
from sklearn.manifold import TSNE, MDS, Isomap
from scipy.spatial.distance import cosine, euclidean
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


def pairwise_metric(df, metric, min_periods=1):
    mat = df.as_matrix().T
    K = len(df.columns)
    met = np.empty((K, K), dtype=float)
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
                c = kappa(ac[valid], bc[valid])
            else:
                c = kappa(ac, bc)
            met[i, j] = c
            met[j, i] = c
    return remove_nans(pd.DataFrame(met, index=df.columns, columns=df.columns))


def similarity_pearson(data):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return remove_nans(data.corr())


def similarity_kappa(data):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, kappa)


def similarity_cosine(data):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, cosine)


def similarity_euclidean(data):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, euclidean)


def similarity_double_pearson(answers):
    return similarity_pearson(similarity_pearson(answers))


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
