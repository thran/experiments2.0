import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
import pandas as pd
import os
import matplotlib.lines as mlines
import matplotlib.pylab as plt
from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse


def sample(answers, n=None, ratio=1):
    if n is not None:
        return answers.sample(n=n)
    return answers.sample(n=int(ratio * len(answers)))

# data_set, n_clusters  = 'matmat-numbers', 3
# data_set, n_clusters  = 'matmat-all', 4
# data_set, n_clusters  = 'simulated-s100-c5-i20', 5
# data_set, n_clusters  = 'simulated-s50-c5-i20', 5
# data_set, n_clusters  = 'simulated-s250-c2-i20', 2
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-addition', 1
# data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2

su = None
for x, (data_set, _) in enumerate([
    # ('cestina-konc-prid', 2),
    # ('cestina-B', 2),
    # ('matmat-numbers', 4),
    # ('matmat-addition', 4),
    # ('matmat-all', 3),
    # ('math_garden-addition', 1),
    # ('math_garden-multiplication', 1),
    ('simulated-s100-c5-i20', 5)
    ]):

    answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    items = pd.read_pickle('data/{}-items.pd'.format(data_set))
    true_cluster_names = list(items['concept'].unique())
    print(true_cluster_names)
    # similarity = similarity_double_pearson
    similarities = [
        # (lambda x: similarity_pearson(x), False, 'Pearson'),
        # (lambda x: similarity_kappa(x), False, 'Cohen'),
        # (lambda x: similarity_yulesQ(x), False, 'Yule'),
        # (lambda x: similarity_ochiai(x), False, 'Ochiai'),
        (lambda x: similarity_jaccard(x), False, 'Jaccard'),
        (lambda x: similarity_sokal(x), False, 'Sokal'),

        # (lambda x: similarity_sokal(x), False, '-'),
        # (lambda x: similarity_pearson(similarity_sokal(x)), False, 'Pearson'),
        # (lambda x: similarity_sokal(x), True, 'Euclid'),
        # (lambda x: similarity_links(similarity_pearson(x)), False, 'Links'),
        # (lambda x: similarity_sokal(x), False, 'Sokal'),
        # (lambda x: similarity_pearson(similarity_sokal(x)), False, 'Sokal -> Pearson'),
        # (lambda x: similarity_sokal(x), True, 'Sokal -> Euclid'),
        # (lambda x: similarity_kappa(x), True, 'Cohen -> Euclid'),
        # (lambda x: similarity_yulesQ(x), True, 'Yule -> Euclid'),

        # (lambda x: similarity_ochiai(x), True, 'Ochiai -> Euclid'),
        # (lambda x: similarity_jaccard(x), True, 'Jaccard -> Euclid'),

        # (lambda x: similarity_pearson(similarity_kappa(x)), False, 'Cohen -> Pearson'),
        # (lambda x: similarity_pearson(similarity_yulesQ(x)), False, 'Yule -> Pearson'),
        # (lambda x: similarity_pearson(similarity_ochiai(x)), False, 'Ochiai -> Pearson'),
        # (lambda x: similarity_pearson(similarity_jaccard(x)), False, 'Jaccard -> Pearson'),
        # (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
        # (lambda x: similarity_pearson(similarity_pearson(x)), False, 'pearson -> pearson'),
        # (lambda x: similarity_links(similarity_pearson(x), 0.1), False, 'pearson -> 0.1 links'),
        # (lambda x: similarity_links(similarity_pearson(x), 0.05), False, 'pearson -> 0.05 links'),
        # (lambda x: similarity_links(similarity_pearson(x)), False, 'pearson -> 0.05 links'),
        # (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
    ]

    # plt.figure(figsize=(16, 5))
    sim = pd.DataFrame()
    for i, (similarity, euclid, similarity_name) in enumerate(similarities):
        print(similarity_name)
        X = similarity(answers)
        if euclid:
            X = -similarity_euclidean(X)
        s = X.as_matrix().flatten()
        n = len(X)
        for j in range(n):  # remove self similarities
            s[j + j * n] = np.nan
        s = s[~np.isnan(s)]
        sim[similarity_name] = s

        items_ids = X.index
        ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])

        plt.subplot(2, len(similarities) // 2, i + 1)
        plt.title(similarity_name)
        plot_similarity_hist(X, ground_truth, similarity_name)


    # sns.pairplot(sim)
    print(sim.corr(method='spearman'))
    # plt.subplot(2, 3, x + 1)
    # plt.title(data_set)
    corr = sim.corr()
    # sns.heatmap(corr, vmax=1, annot=True, vmin=0)

    if su is None:
        su = corr
    else:
        su += corr

# plt.figure()
# sns.heatmap(su / su.max(), annot=True, vmin=0)

    # plt.yticks(rotation=0)

plt.show()
