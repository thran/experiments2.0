import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA

from algorithms.kmean_precomputed import KMeanPrecomputed
from algorithms.spectralclustering import SpectralClusterer
import matplotlib.pylab as plt
from cross_system.clustering.similarity import similarity_euclidean

colors = "rgbykcmw"
markers = "o^s*vx+"


def spectral_clustering(similarity, concepts=2, euclid=False):
    if euclid:
        X = similarity_euclidean(similarity)
    else:
        X = similarity
        X[X < 0] = 0
    sc = SpectralClusterer(X, kcut=X.shape[0] / 2, mutual=True)
    return sc.run(cluster_number=concepts, KMiter=50, sc_type=2)


def spectral_clustering2(similarity, concepts=2, euclid=False):
    if euclid:
        model = SpectralClustering(n_clusters=concepts, affinity='nearest_neighbors')
        return model.fit_predict(similarity)
    else:
        model = SpectralClustering(n_clusters=concepts, affinity='precomputed')
        similarity[similarity < 0] = 0
        return model.fit_predict(similarity)


def hierarchical(similarity, concepts=2, euclid=False):
    if euclid:
        model = AgglomerativeClustering(n_clusters=concepts)
        return model.fit_predict(similarity)
    else:
        model = AgglomerativeClustering(n_clusters=concepts, affinity='precomputed', linkage='complete')
        return model.fit_predict(1 - similarity)


def kmeans(similarity, concepts=2, euclid=False):
    if euclid:
        kmeans = KMeans(n_clusters=concepts, n_init=100, max_iter=1000)
        return kmeans.fit_predict(similarity)
    else:
        return kmeans_precomputed(similarity, concepts=concepts)


def kmeans_precomputed(similarity, concepts=2):
    X = 1 - similarity.as_matrix()
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


def dbscan(similarity, concepts=2, euclid=False):
    if euclid:
        model = DBSCAN(eps=0.6, min_samples=10, algorithm='auto', leaf_size=30)
        return model.fit_predict(similarity)
    else:
        model = DBSCAN(eps=0.6, min_samples=10, metric='precomputed', algorithm='auto', leaf_size=30)
        return model.fit_predict(1 - similarity)


def plot_clustering(items, xs, ys, labels=None, texts=None, shapes=None):
    labels = [0] * len(xs) if labels is None else labels
    texts = [''] * len(xs) if texts is None else texts
    shapes = [0] * len(xs) if shapes is None else shapes
    for item, x, y, label, text, shape in zip(items, xs, ys, labels, texts, shapes):
        plt.plot(x, y, markers[shape], color=colors[label])
        plt.text(x, y, text)


