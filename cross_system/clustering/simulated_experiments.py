import math
import random
import seaborn as sns
import numpy as np
import pandas as pd
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
from sklearn.metrics import adjusted_rand_score as rand_index

def sigmoid(x, c=0):
    return c + (1 - c) / (1 + math.exp(-x))

def data(n_students=100, n_concepts=5, n_items=20, difficulty_shift=0.5, skill_correlation=0):

    skill_correlation_matrix = np.zeros((n_concepts, n_concepts))
    skill_correlation_matrix.fill(skill_correlation)
    np.fill_diagonal(skill_correlation_matrix, 1)
    skill_correlation_matrix /= skill_correlation_matrix.sum(axis=1)
    skill = np.dot(np.random.randn(n_students, n_concepts), skill_correlation_matrix) + np.array([-1, 1])
    items = np.array([[i,  i // n_items] for i in range(n_concepts * n_items)])
    difficulty = np.random.randn(len(items)) - difficulty_shift # shift to change overall difficulty

    answers = []
    for s in range(n_students):
        for i, concept in items:
            prob = sigmoid(skill[s, concept] - difficulty[i])
            answers.append([s, i, 0, 1 *(random.random() < prob) ])

    answers = pd.DataFrame(answers, columns=['student', 'item', 'response_time', 'correct'])
    items = pd.DataFrame(items, columns=['name', 'concept'])

    return answers, items


similarities = [
    (lambda x: similarity_pearson(x), False, 'pearson'),
    (lambda x: similarity_yulesQ(x), False, 'yuleQ'),
    (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
    (lambda x: similarity_kappa(x), True, 'kappa -> euclid'),
    (lambda x: similarity_yulesQ(x), True, 'yuleQ -> euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
]
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

similarity, euclid, similarity_name = similarities[2]
n_clusters = 2
n_students = 50
skill_correlation = 0
difficulty_shift = 0.5

runs = 10
results = []
if True:
    for run in range(runs):
        print(run)
        for n_students in range(10, 121, 10):
        # for difficulty_shift in np.arange(-1, 1.1, 0.2):
            answers, items = data(n_students=n_students, n_concepts=n_clusters, skill_correlation=skill_correlation, difficulty_shift=difficulty_shift)

            true_cluster_names = list(items['concept'].unique())
            X = similarity(answers)
            items_ids = X.index
            ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

            for i, clustering in enumerate(clusterings):

                labels = clustering(X, n_clusters, euclid=euclid)
                rand = rand_index(ground_truth, labels)
                results.append([n_students, clustering.__name__, rand, skill_correlation, difficulty_shift])

if False:
    clustering = kmeans
    for run in range(runs):
        print(run)
        for skill_correlation in np.arange(0, 1.1, 0.1):
            print(skill_correlation)
            for n_students in (100, 1000, 10000):
                answers, items = data(n_students=n_students, n_concepts=n_clusters, skill_correlation=skill_correlation)

                true_cluster_names = list(items['concept'].unique())
                X = similarity(answers)
                items_ids = X.index
                ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

                labels = clustering(X, n_clusters, euclid=euclid)
                rand = rand_index(ground_truth, labels)
                results.append([n_students, clustering.__name__, rand, skill_correlation, difficulty_shift])

results = pd.DataFrame(results, columns=['students', 'clustering', 'rand_index', 'skill_correlation', 'difficulty_shift'])
print(results)

plt.figure(figsize=(16, 24))
plt.title(similarity_name)
sns.pointplot(data=results, x='students', y='rand_index', hue='clustering')

plt.show()