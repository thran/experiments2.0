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
data_set, n_clusters  = 'simulated-s50-c5-i20', 5
# data_set, n_clusters  = 'simulated-s250-c2-i20', 2
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-addition', 1
# data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2
# data_set, n_clusters = 'cestina-konc-prid', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())
print(true_cluster_names)
# similarity = similarity_double_pearson
similarities = [
    (lambda x: similarity_pearson(x), False, 'pearson'),
    (lambda x: similarity_pearson(x), True, 'pearson -> euclid'),
    (lambda x: similarity_pearson(similarity_pearson(x)), False, 'pearson -> pearson'),
    (lambda x: similarity_links(similarity_pearson(x), 0.1), False, 'pearson -> 0.1 links'),
    (lambda x: similarity_links(similarity_pearson(x), 0.05), False, 'pearson -> 0.05 links'),
    (lambda x: similarity_pearson(similarity_pearson(x)), True, 'pearson -> pearson -> euclid'),
]

plt.figure(figsize=(16, 5))
plt.suptitle(data_set)
for i, (similarity, euclid, similarity_name) in enumerate(similarities):
    print(similarity_name)
    X = similarity(answers)
    if euclid:
        X = similarity_euclidean(X)
    items_ids = X.index
    ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
    same, different = [], []
    for concept1 in set(ground_truth):
        for concept2 in set(ground_truth):
            values = list(X.loc[ground_truth == concept1, ground_truth == concept2].values.flatten())
            if concept1 == concept2:
                same += values
            elif concept1 > concept2:
                different += values

    plt.subplot(1, len(similarities), i + 1)
    plt.title(similarity_name)
    if similarity_name.endswith('links'):
        sns.distplot(same)
        if len(different):
            sns.distplot(different)
    elif not euclid:
        plt.xlim([-1,1])
        sns.distplot(same)
        if len(different):
            sns.distplot(different)
    else:
        if len(different):
            plt.xlim([-max(different), 0])
            sns.distplot(-np.array(different))
        sns.distplot(-np.array(same))



plt.show()
