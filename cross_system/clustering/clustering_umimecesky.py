import math
import random
import seaborn as sns
import numpy as np
import pandas as pd
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
from sklearn.metrics import adjusted_rand_score as rand_index

# data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2
data_set, n_clusters = 'cestina-konc-prid', 7
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())
n_clusters = len(true_cluster_names)
print(true_cluster_names)

similarities = [
    (lambda x: similarity_pearson(x), False, 'pearson'),
    # (lambda x: similarity_yulesQ(x), False, 'yuleQ'),
    # (lambda x: similarity_kappa(x), False, 'kappa'),
    # (lambda x: similarity_cosine(x), False, 'cosine'),
    # (lambda x: similarity_euclidean(x), False, 'euclid'),
    # (lambda x: similarity_kappa(x), True, 'kappa -> euclid'),
    (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
    # (lambda x: similarity_yulesQ(x), True, 'yuleQ -> euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), False, 'pearson -> pearson'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), False, 'yuleQ -> pearson'),
    # (lambda x: similarity_links(similarity_yulesQ(x)), False, 'yuleQ -> links'),
    (lambda x: similarity_links(similarity_pearson(x)), False, 'pearson -> links'),
    (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), True, 'yuleQ -> pearson -> euclid'),
    # (lambda x: similarity_euclidean(similarity_pearson(x)), True, 'pearson -> euclid -> euclid'),
]
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

print(len(answers))
clustering = hierarchical

results = []
for run in range(50):
    for points in range(10000, 60001, 10000):
        for similarity, euclid, similarity_name in similarities:
            print(run, points, similarity_name)
            X = similarity(answers.sample(n=points))
            items_ids = X.index
            ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

            labels = clustering(X, n_clusters, euclid=euclid)
            rand = rand_index(ground_truth, labels)
            results.append([points, clustering.__name__, rand, similarity_name])

results = pd.DataFrame(results, columns=['points', 'clustering', 'rand_index', 'similarity'])
print(results)

plt.figure(figsize=(16, 24))
sns.pointplot(data=results, x='points', y='rand_index', hue='similarity')


plt.show()