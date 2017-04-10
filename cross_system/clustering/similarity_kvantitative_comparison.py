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

# data_set, n_clusters  = 'simulated-s100-c5-i20', 5
# data_set, n_clusters  = 'simulated-s250-c2-i20', 2
# data_set, n_clusters = 'cestina-konc-prid', 7

similarities = [
    (lambda x: similarity_pearson(x), False, 'Pearson'),
    (lambda x: similarity_jaccard(x), False, 'Jaccard'),
    # (lambda x: similarity_kappa(x), False, 'Cohen'),
    (lambda x: similarity_yulesQ(x), False, 'Yule'),
    # (lambda x: similarity_ochiai(x), False, 'Ochiai'),
    (lambda x: similarity_sokal(x), False, 'Sokal'),
    (lambda x: similarity_pearson(x), True, 'Pearson -> Euclid'),
    # (lambda x: similarity_kappa(x), True, 'Cohen -> Euclid'),
    # (lambda x: similarity_ochiai(x), True, 'Ochiai -> Euclid'),
    # (lambda x: similarity_sokal(x), True, 'Sokal -> Euclid'),
    (lambda x: similarity_jaccard(x), True, 'Jaccard -> Euclid'),
    (lambda x: similarity_yulesQ(x), True, 'Yule -> Euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), False, 'Pearson -> Pearson'),
    # (lambda x: similarity_pearson(similarity_kappa(x)), False, 'Cohen -> Pearson'),
    (lambda x: similarity_pearson(similarity_jaccard(x)), False, 'Jaccard -> Pearson'),
    (lambda x: similarity_pearson(similarity_yulesQ(x)), False, 'Yule -> Pearson'),
    # (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
    # (lambda x: similarity_pearson(similarity_pearson(x)), False, 'pearson -> pearson'),
    # (lambda x: similarity_links(similarity_pearson(x), 0.1), False, 'pearson -> 0.1 links'),
    # (lambda x: similarity_links(similarity_pearson(x), 0.05), False, 'pearson -> 0.05 links'),
    # (lambda x: similarity_links(similarity_pearson(x)), False, 'pearson -> 0.05 links'),
    # (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
]

clustering = hierarchical

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


for n_students, n_clusters in [
    (50, 5),
    # (100, 5),
    # (200, 5),
    # (100, 2),
    # (100, 10),
]:
    results = defaultdict(lambda: [])
    for run in range(20):
        print(run)
        # answers, items = data(n_students=n_students, n_concepts=n_clusters, n_items=20)
        data_set, n_clusters = 'cestina-konc-prid', 7
        answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
        items = pd.read_pickle('data/{}-items.pd'.format(data_set))

        students = pd.Series(answers['student'].unique())
        answers = answers[answers['student'].isin(students.sample(n=1100))]

        true_cluster_names = list(items['concept'].unique())

        for similarity, euclid, similarity_name in similarities:
            X = similarity(answers)
            if euclid:
                X = similarity_euclidean(X)
            items_ids = X.index
            ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])

            labels = clustering(X, n_clusters, euclid=euclid)
            rand = rand_index(ground_truth, labels)
            results[similarity_name].append(rand)


    print()
    print()
    print(n_students, n_clusters)
    for similarity, euclid, similarity_name in similarities:
        rands = np.array(results[similarity_name])
        m, h = mean_confidence_interval(rands)
        print(similarity_name, round(m, 2), round(h, 2))

plt.show()
