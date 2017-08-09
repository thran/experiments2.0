from sklearn.decomposition import PCA
from algorithms.spectralclustering import SpectralClusterer
from sklearn.manifold import TSNE, MDS, Isomap

from cross_system.clustering.similarity import similarity_euclidean


def spectral(similarity, euclid=False):
    if euclid and False:
        similarity = similarity_euclidean(similarity)
    else:
        similarity[similarity < 0] = 0
    sc = SpectralClusterer(similarity, kcut=similarity.shape[0] / 2, mutual=True)

    sc.run(cluster_number=2, KMiter=50, sc_type=2)
    return (sc.eig_vect[:, 1], sc.eig_vect[:, 2])


def pca(similarity, n_components=2, euclid=False):
    if not euclid:
        print('podvod')
    model = PCA(n_components=n_components)
    result = model.fit_transform(similarity)

    return result.T


def tsne(similarity, euclid=False, perplexity=30):
    if euclid:
        model = TSNE(learning_rate=100, perplexity=perplexity, n_iter=200000)
        result = model.fit_transform(similarity)
    else:
        model = TSNE(learning_rate=100, perplexity=perplexity, n_iter=100000, init='random', metric='precomputed')
        result = model.fit_transform(1 - similarity)

    return result.T


def isomap(similarity, euclid=False):
    if not euclid:
        print('podvod')
    model = Isomap(n_neighbors=15)
    result = model.fit_transform(similarity)

    return result.T


def mds(similarity, euclid=False):
    if euclid:
        model = MDS(max_iter=1000)
        result = model.fit_transform(similarity)
    else:
        model = MDS(max_iter=1000, dissimilarity='precomputed')
        result = model.fit_transform(1 - similarity)

    return result.T
