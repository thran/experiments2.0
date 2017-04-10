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
# data_set, n_clusters  = 'simulated-s100-c2-i50', 2
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-multiplication', 1
# data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2
data_set, n_clusters = 'cestina-konc-prid', 7
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

# similarity = similarity_double_pearson
similarities = [
    (lambda x: similarity_pearson(x), False, 'pearson'),
    (lambda x: similarity_yulesQ(x), False, 'yuleQ'),
    (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
    # (lambda x: similarity_kappa(x), True, 'kappa -> euclid'),
    (lambda x: similarity_yulesQ(x), True, 'yuleQ -> euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
]
dimensions = 0
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

runs = 1
results = []
for run in range(runs):
    A = answers.sample(frac=0.5)
    for similarity, euclid, similarity_name in similarities:
        print(similarity_name)
        X = similarity(A)
        items_ids = X.index
        if dimensions:
            model = PCA(n_components=dimensions)
            X = pd.DataFrame(data=model.fit_transform(X), index=X.index)

        ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

        for i, clustering in enumerate(clusterings):

            labels = clustering(X, n_clusters, euclid=euclid)
            rand = rand_index(ground_truth, labels)
            print('  - ', clustering.__name__, rand)
            results.append([similarity_name, clustering.__name__, rand])

results = pd.DataFrame(results, columns=['similarity', 'clustering', 'rand_index'])
print(results)

plt.figure(figsize=(16, 24))
plt.title(data_set)
sns.barplot(data=results, x='similarity', y='rand_index', hue='clustering')

plt.show()
