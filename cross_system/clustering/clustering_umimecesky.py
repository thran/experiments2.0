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
    (lambda x: similarity_pearson(x), False, 'Pearson'),
    # (lambda x: similarity_kappa(x), False, 'kappa'),
    # (lambda x: similarity_cosine(x), False, 'Ochiai'),
    # (lambda x: similarity_yulesQ(x), False, 'Yule'),
    # (lambda x: similarity_accuracy(x), False, 'accuracy'),
    # (lambda x: similarity_jaccard(x), False, 'Jaccard'),
    # (lambda x: similarity_euclidean(x), False, 'euclid'),
    (lambda x: similarity_pearson(x), True, 'Pearson -> Euclid'),
    # (lambda x: similarity_links(similarity_pearson(x), 0.1), True, 'pearson -> links -> euclid'),
    # (lambda x: similarity_kappa(x), True, 'kappa -> euclid'),
    # (lambda x: similarity_yulesQ(x), True, 'Yule -> Euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), False, 'Pearson -> Pearson'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), False, 'Yule -> Pearson'),
    # (lambda x: similarity_links(similarity_yulesQ(x)), False, 'Yule -> Links'),
    # (lambda x: similarity_links(similarity_pearson(x)), False, 'Pearson -> Links'),
    (lambda x: similarity_pearson(similarity_pearson(x)), True, 'Pearson -> Pearson -> Euclid'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), True, 'Yule -> Pearson -> Euclid'),
    # (lambda x: similarity_euclidean(similarity_pearson(x)), True, 'pearson -> euclid -> euclid'),
]
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

print(len(answers))
clustering = hierarchical
students = pd.Series(answers['student'].unique())
print(len(students))
results = []
for run in range(50):
    for student_count in list(range(100, 1101, 100)) + [len(students)]:
    # for student_count in range(10000, 60001, 10000):
        A = answers[answers['student'].isin(students.sample(n=student_count))]
        # A = answers.sample(n=student_count)
        print(run, student_count, len(A))
        for similarity, euclid, similarity_name in similarities:
            X = similarity(A)
            items_ids = X.index
            ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

            labels = clustering(X, n_clusters, euclid=euclid)
            rand = rand_index(ground_truth, labels)
            results.append([student_count, clustering.__name__, rand, similarity_name])

results = pd.DataFrame(results, columns=['students', 'clustering', 'rand_index', 'similarity'])


plt.figure(figsize=(16, 24))
sns.pointplot(data=results, x='students', y='rand_index', hue='similarity')
# sns.violinplot(data=results, x='students', y='rand_index', hue='similarity')


plt.show()