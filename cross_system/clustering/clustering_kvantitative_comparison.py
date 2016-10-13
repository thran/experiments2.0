import random
from collections import defaultdict

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index
import scipy.stats
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
import pandas as pd
import os
import matplotlib.lines as mlines
import matplotlib.pylab as plt

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

def data(n_students=100, n_concepts=5, n_items=15, difficulty_shift=0.5, skill_correlation=0., missing=0.):
    def sigmoid(x, c=0):
        return c + (1 - c) / (1 + math.exp(-x))

    skill_correlation_matrix = np.zeros((n_concepts, n_concepts))
    skill_correlation_matrix.fill(skill_correlation)
    np.fill_diagonal(skill_correlation_matrix, 1)
    skill = np.random.multivariate_normal(np.zeros(n_concepts), skill_correlation_matrix, n_students)
    items = np.array([[i,  i // n_items] for i in range(n_concepts * n_items)])
    difficulty = np.random.randn(len(items)) - difficulty_shift # shift to change overall difficulty

    answers = []
    for s in range(n_students):
        for i, concept in items:
            if random.random() < missing:
                continue
            prob = sigmoid(skill[s, concept] - difficulty[i])
            answers.append([s, i, 0, 1 *(random.random() < prob) ])

    answers = pd.DataFrame(answers, columns=['student', 'item', 'response_time', 'correct'])
    items = pd.DataFrame(items, columns=['name', 'concept'])

    return answers, items


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


similarity, euclid, similarity_name = similarity_yulesQ, True, 'pearson -> euclid'
clusterings = [kmeans, hierarchical, spectral_clustering2]

results = defaultdict(lambda: [])
for run in range(50):
    print(run)
    (answers, items), n_clusters = data(n_students=50, n_concepts=5, n_items=20), 5
    # (answers, items), n_clusters = data(n_students=50, n_concepts=2, n_items=50), 2
    # data_set, n_clusters = 'cestina-konc-prid', 7
    # data_set, n_clusters = 'matmat-numbers', 3
    # answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    # items = pd.read_pickle('data/{}-items.pd'.format(data_set))

    true_cluster_names = list(items['concept'].unique())
    X = similarity(answers)
    for clustering in clusterings:
        items_ids = X.index
        ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])

        labels = clustering(X, n_clusters, euclid=euclid)
        rand = rand_index(ground_truth, labels)
        results[clustering.__name__].append(rand)


print('')
for clustering in clusterings:
    rands = np.array(results[clustering.__name__])
    m, h = mean_confidence_interval(rands)
    print('& ${} \pm {}$'.format(round(m, 2), round(h, 2)), end=' ')
    # print(clustering.__name__, round(m, 2), round(h, 2))

plt.show()
