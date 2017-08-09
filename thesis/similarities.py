import json
from itertools import permutations, combinations

import pandas as pd
import numpy as np
import seaborn as sns

from cross_system.clustering.projection import tsne
from cross_system.clustering.similarity import *
from cross_system.clustering.clusterings import *
from thesis.data.tutor.process import problems
from sklearn.metrics import adjusted_rand_score as rand_index

problem = problems[0]
problem1, problem2 = problems[3], problems[7]

def acc(labels1, labels2):
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    results = []
    for perm in permutations(set(labels1)):
        permuted_labels = [perm[l] for l in labels1]
        results.append((permuted_labels == labels2).sum() / len(labels1))

    return max(results)


def get_data(problem):
    df = pd.read_pickle('tutor/{}.pd'.format(problem))
    df = np.log(df)
    return df

similarities = [
    (lambda x: similarity_pearson(x), 'Pearson'),
    (lambda x: similarity_pearson(similarity_pearson(x)), 'Pearson -> Pearson'),
    (lambda x: similarity_euclidean(x), 'Euclid'),
    (lambda x: similarity_cosine(x), 'Cosine'),
]


if False:
    if True:
        df1 = get_data(problem1)
        df2 = get_data(problem2)
        labels = np.array([0] * len(df1.columns) + [1] * len(df2.columns))

        df = pd.concat([df1, df2], axis=1)
    else:
        dfs = [get_data(problem) for problem in problems]
        labels = []
        for i, df in enumerate(dfs):
            labels += [i] * len(df.columns)
        df = pd.concat(dfs, axis=1)
        labels = np.array(labels)

    for i,(similarity, name) in enumerate(similarities):
        plt.subplot(2, 2, i + 1)
        plt.title(name)
        plot_similarity_hist(similarity(df), labels, name, False)


def cluster(problems, clustering):
    dfs = [get_data(problem) for problem in problems]
    labels = []
    for i, df in enumerate(dfs):
        labels += [i] * len(df.columns)
    df = pd.concat(dfs, axis=1)
    labels = np.array(labels)

    sim = similarity_pearson(similarity_pearson(df))

    results = clustering(sim, len(problems))
    return rand_index(results, labels), acc(results, labels)

if False:
    n = 10
    l1 = [0] * (n // 5) + [0]*n + [1]*n
    l2 = [1] * (n // 5) + [0]*n + [1]*n
    print(rand_index(l1, l2))
    print(acc(l1, l2))

if False:
    for clustering in [kmeans, hierarchical, spectral_clustering2]:
        rands = []
        accs = []
        for combination in combinations(problems, 5):
            results = cluster(combination, clustering)
            # print(i, j, clustering.__name__, results)
            rands.append(results[0])
            accs.append(results[1])
        print(clustering.__name__, np.array(rands).mean(), np.array(accs).mean())

if True:
    answers = pd.read_pickle('data/slepemapy/answers.pd')
    items = json.load(open('data/slepemapy/Europe.json'))
    answers = answers[answers['item'].isin(map(int, items.keys()))]
    answers = answers.groupby(['student', 'item']).first().reset_index()
    answers['correct'] = answers['correct'].astype(int)

    print(len(answers))

    X = similarity_pearson(similarity_pearson(answers))
    xs, ys = tsne(X, euclid=False, perplexity=5)
    labels = spectral_clustering2(X, 3)
    for label in set(labels):
        print(label)
        for l, id in zip(labels, X.columns):
            if l == label:
                print(items[str(id)], end=", ")
        print('')

    plot_clustering(
        X.index, xs, ys,
        # labels=ground_truth,
        texts=[items[str(item)] for item in X.index],
        shapes=None,
    )


plt.show()