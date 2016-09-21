#!/usr/bin/python3

import numpy as np
import random
import math
import pylab as plt
from scipy.cluster.vq import kmeans2
from sklearn.metrics import adjusted_rand_score
from scipy.stats import pearsonr, spearmanr
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class SimData:
    def __init__(self, students = 100, concepts = 5, items = 20):
        self.students = students
        self.concepts = concepts
        self.items = items
        self.skill = np.random.randn(students, concepts)*0.7 # hack - adjust skill variability
        self.difficulty = np.random.randn(concepts * items) - 0.5 # hack - shift to change overall difficulty
        self.item_concept = np.array([ i // items for i in range(concepts*items) ])
        self.answers = np.zeros((students, concepts * items), dtype=np.int)
        for s in range(students):
            for i in range(concepts * items):
                prob = sigmoid(self.skill[s, self.item_concept[i]] - self.difficulty[i])
                if random.random() < prob:
                    self.answers[s, i] = 1

def quality_evolution_test(color = "blue", rep = 20, double_cor = False):
    sizes, vals = [], []
    for s in [10, 25, 50, 100, 200, 300,  400, 600, 1000]:
        yvals = []
        for _ in range(rep):
            data = SimData(s, 4, 15)
            cor = np.nan_to_num(np.corrcoef(data.answers, rowvar=0)) # pearson metric
            if double_cor:
                cor = np.nan_to_num(np.corrcoef(cor))

            label = kmeans2(cor, 6, minit='points', iter=100, thresh=1e-6)[1] # hack pocet komponent
            # label = kmeans(cor, 4, True)
            alg_labels = [ label[i] for i in range(data.items * data.concepts) ]
            truth = data.item_concept
            y = adjusted_rand_score(truth, alg_labels)
            print(s, y)
            plt.scatter(s, y, color=color)
            yvals.append(y)
        sizes.append(s)
        vals.append(np.mean(yvals))
    plt.plot(sizes, vals, color=color)

def comparison():
    quality_evolution_test("blue")
    quality_evolution_test("red",double_cor=True)
    plt.xlabel("students")
    plt.ylabel("Adj. Rand index")
    plt.legend(("pearson", "pearson->pearson"), loc = "best")
    plt.show()

comparison()

def test():
    data = SimData(400, 4, 15)
    cor = np.nan_to_num(np.corrcoef(data.answers, rowvar=0)) # pearson metric
    cor = np.nan_to_num(np.corrcoef(cor))
    label1 = kmeans2(cor, 6, minit='points', iter=100)[1] # hack pocet komponent
    label2 = kmeans(cor, 6, True)

    xs, ys = mds(cor, euclid=True)
    plt.subplot(1, 2, 1)
    plt.title('kmeans2 ' + str(adjusted_rand_score(data.item_concept, label1)))
    plot_clustering(
        range(cor.shape[0]), xs, ys,
        labels=label1,
        shapes=data.item_concept,
    )

    plt.subplot(1, 2, 2)
    plt.title('Kmeans ' + str(adjusted_rand_score(data.item_concept, label2)))
    plot_clustering(
        range(cor.shape[0]), xs, ys,
        labels=label2,
        shapes=data.item_concept,
    )

    plt.show()