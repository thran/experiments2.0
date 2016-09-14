import numpy as np
from sklearn.cluster import KMeans

from algorithms.kmean_precomputed import KMeanPrecomputed
from algorithms.spectralclustering import SpectralClusterer
import matplotlib.pylab as plt


colors = "rgbyk"
markers = "o^s*vx"


def spectral_clustering(X, concepts=2):
    X[X < 0] = 0
    sc = SpectralClusterer(X, kcut=X.shape[0] / 2, mutual=True)
    return sc.run(cluster_number=concepts, KMiter=50, sc_type=2)



def kmeans(X, concepts=2):
    kmeans = KMeans(n_clusters=concepts, n_init=100, max_iter=1000)
    return kmeans.fit_predict(X)


def kmeans_precomputed(X, concepts=2):
    kmeans = KMeanPrecomputed(X)
    best = np.inf
    labels = None
    for i in range(100):
        l = kmeans.run(concepts)
        cost = kmeans.cluster_cost()
        if cost < best:
            labels = l
            best = cost
    return labels


def plot_clustering(items, xs, ys, labels=None, texts=None, shapes=None):
    labels = [0] * len(xs) if labels is None else labels
    texts = [""] * len(xs) if texts is None else texts
    shapes = [0] * len(xs) if shapes is None else shapes
    for item, x, y, label, text, shape in zip(items, xs, ys, labels, texts, shapes):
        plt.plot(x, y, markers[shape], color=colors[label])
        plt.text(x, y, text)


