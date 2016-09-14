from sklearn.decomposition import PCA
from algorithms.spectralclustering import SpectralClusterer
from sklearn.manifold import TSNE, MDS, Isomap


def spectral_clustering(X):
    X[X < 0] = 0
    sc = SpectralClusterer(X, kcut=X.shape[0] / 2, mutual=True)

    sc.run(cluster_number=2, KMiter=50, sc_type=2)
    return (sc.eig_vect[:, 1], sc.eig_vect[:, 2])


def tsne(X):
    model = TSNE(learning_rate=300, n_iter=100000, init='pca')
    result = model.fit_transform(X)

    return result.T


def pca(X, n_components=2):
    model = PCA(n_components=n_components)
    result = model.fit_transform(X)

    return result.T


def isomap(X):
    model = Isomap(n_neighbors=15)
    result = model.fit_transform(X)

    return result.T


def mds(X):
    model = MDS(max_iter=1000)
    result = model.fit_transform(X)

    return result.T
