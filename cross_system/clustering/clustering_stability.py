from collections import defaultdict

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
data_set, n_clusters = 'cestina-Z', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

# similarity_setting = lambda x: similarity_pearson(x), False, 'pearson'
# similarity_setting = lambda x: similarity_yulesQ(x), False, 'yuleQ'
similarity_setting = lambda x: similarity_pearson(x), True, 'pearson -> euclid'
# similarity_setting = lambda x: similarity_yulesQ(x), True, 'yuleQ -> euclid'
# similarity_setting = lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'

dimensions = 0
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

runs = 10
results = []
inner_results = []
for frac in np.arange(0.1, 1.1, 0.1):
    all_labels = defaultdict(lambda: [])
    for run in range(runs):
        A = answers.sample(frac=frac)
        similarity, euclid, similarity_name = similarity_setting
        X = similarity(A)
        items_ids = X.index
        if dimensions:
            model = PCA(n_components=dimensions)
            X = pd.DataFrame(data=model.fit_transform(X), index=X.index)

        ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

        for clustering in clusterings:
            labels = clustering(X, n_clusters, euclid=euclid)
            all_labels[clustering].append(labels)
            rand = rand_index(ground_truth, labels)
            print('  - ', clustering.__name__, rand)
            results.append([frac, clustering.__name__, rand])
    for clustering in clusterings:
        for i, l1 in enumerate(all_labels[clustering]):
            for j, l2 in enumerate(all_labels[clustering]):
                if i <= j:
                    continue
                rand = rand_index(l1, l2)
                inner_results.append([frac, clustering.__name__, rand])

results = pd.DataFrame(results, columns=['frac', 'clustering', 'rand_index'])
inner_results = pd.DataFrame(inner_results, columns=['frac', 'clustering', 'rand_index'])
print(results)

plt.figure(figsize=(16, 24))
plt.title('{} - {}'.format(data_set, similarity_setting[2]))
sns.pointplot(data=results, x='frac', y='rand_index', hue='clustering')

plt.figure(figsize=(16, 24))
plt.title('{} - {}'.format(data_set, similarity_setting[2]))
sns.pointplot(data=inner_results, x='frac', y='rand_index', hue='clustering')

plt.show()
