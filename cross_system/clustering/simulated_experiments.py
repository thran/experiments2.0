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

def data(n_students=100, n_concepts=5, n_items=15, difficulty_shift=0.5, skill_correlation=0., missing=0.):

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


similarities = [
    # (lambda x: similarity_pearson(x), False, 'Pearson'),
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
    # (lambda x: similarity_pearson(similarity_pearson(x)), False, 'Pearson -> Pearson'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), False, 'Yule -> Pearson'),
    # (lambda x: similarity_links(similarity_yulesQ(x)), False, 'Yule -> Links'),
    # (lambda x: similarity_links(similarity_pearson(x)), False, 'Pearson -> Links'),
    # (lambda x: similarity_pearson(similarity_pearson(x)), True, 'Pearson -> Pearson -> Euclid'),
    # (lambda x: similarity_pearson(similarity_yulesQ(x)), True, 'Yule -> Pearson -> Euclid'),
    # (lambda x: similarity_euclidean(similarity_pearson(x)), True, 'pearson -> euclid -> euclid'),
]
clusterings = [
    kmeans,
    spectral_clustering2,
    hierarchical
]

similarity, euclid, similarity_name = similarities[0]
n_clusters = 5
n_items = 20
n_students = 100
skill_correlation = 0
difficulty_shift = 0.5
clustering = kmeans
missing = 0.



def students(runs=1):
    results = []
    for run in range(runs):
        # for n_students in range(100, 1001, 100):
        # for n_students in [10, 25, 50, 100, 200, 300,  400, 600]:
        for difficulty_shift in np.arange(-1, 1.1, 0.2):
            answers, items = data(n_students=n_students, n_items=n_items, n_concepts=n_clusters, skill_correlation=skill_correlation, difficulty_shift=difficulty_shift, missing=missing)
            true_cluster_names = list(items['concept'].unique())
            # for i, clustering in enumerate(clusterings):
            for similarity, euclid, similarity_name in similarities:
                X = similarity(answers)
                items_ids = X.index
                ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

                labels = clustering(X, n_clusters, euclid=euclid)
                rand = rand_index(ground_truth, labels)
                results.append([n_students, clustering.__name__, rand, skill_correlation, difficulty_shift, similarity_name])
                print(run, n_students, similarity_name, rand)

    results = pd.DataFrame(results, columns=['students', 'clustering', 'rand_index', 'skill_correlation', 'difficulty_shift', 'similarity'])
    print(results)

    plt.figure(figsize=(16, 24))
    sns.pointplot(data=results, x='difficulty_shift', y='rand_index', hue='similarity')


def skill_correlations(runs=50, n_clusters=5):
    results = []
    clustering = kmeans
    for run in range(runs):
        for skill_correlation in list(np.arange(0, 0.9, 0.1)) + [0.85]:
            for clustering in clusterings:
                for students in [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
                    answers, items  = data(n_students=students, n_items=20, n_concepts=n_clusters, skill_correlation=skill_correlation)
                    true_cluster_names = list(items['concept'].unique())
                    X = similarity(answers)
                    items_ids = X.index
                    ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

                    labels = clustering(X, n_clusters, euclid=euclid)
                    rand = rand_index(ground_truth, labels)

                    print(run, skill_correlation, clustering.__name__, students, '===', rand)
                    if rand >= 0.9:
                        results.append([students, clustering.__name__, rand, skill_correlation])
                        break

    results = pd.DataFrame(results, columns=['students', 'clustering', 'rand_index', 'skill_correlation'])

    print(results)
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(yscale="log")
    sns.pointplot(data=results, x='skill_correlation', y='students', hue='clustering', ax=ax)



# skill_correlations()
students()

plt.show()