import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.similarity import *
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse, \
    MathGardenResponseModificator

data_set, n_clusters  = 'matmat-numbers', 3
# data_set, n_clusters  = 'matmat-multiplication', 1
# data_set, n_clusters  = 'matmat-addition', 1
# data_set, n_clusters  = 'matmat-all', 4
# data_set, n_clusters  = 'math_garden-multiplication', 1
# data_set, n_clusters  = 'math_garden-addition', 1
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-all2', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

median = np.round(np.median(answers['response_time']), 1)
print(median)
# modificators = [BinaryResponse(), TimeLimitResponseModificator([(median, 0.5)]), MathGardenResponseModificator(2 * median), LinearDrop(2 * median)]
modificators = [BinaryResponse(), LinearDrop(2 * median)]

similarity, euclid = similarity_pearson, True
projection = mds
clustering = kmeans

if False:
# answers = answers.sample(n=20000)
    print(len(answers))
    data = pd.DataFrame()
    for modificator in modificators:
        print(modificator)
        modified_answers = modificator.modify(answers.copy())
        # print(modified_answers)
        X = similarity(modified_answers)
        data[str(modificator)] = X.as_matrix().flatten()

    data = data.replace([1],0)
    print(data.corr().as_matrix().round(2))
    # print(data.corr(method='spearman'))

    g = sns.pairplot(data, diag_kind="kde")
    g.map_lower(sns.kdeplot, cmap="Blues_d")

plt.figure(figsize=(8, 8))
for i, modificator in enumerate(modificators):
    print(modificator)
    modified_answers = modificator.modify(answers.copy())
    X = similarity(modified_answers)
    xs, ys = projection(X, euclid=euclid)
    ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])

    plt.subplot(2, len(modificators), i + 1)
    plt.title(str(modificator))
    plot_clustering(
        X.index, xs, ys,
        labels=3 + 1 * (np.array([int(items.get_value(item, 'name')) for item in X.index]) > 10),
        texts=[items.get_value(item, 'name') for item in X.index],
        # shapes=ground_truth,
    )

    plt.subplot(2, len(modificators), i + 3)
    plt.title(str(modificator))
    plot_clustering(
        X.index, xs, ys,
        labels=ground_truth,
        texts=[items.get_value(item, 'name') for item in X.index],
        # shapes=np.array([int(items.get_value(item, 'name')) for item in X.index]) > 10,
    )


plt.legend(handles=[
    mlines.Line2D([], [], color=colors[i], linewidth=0, marker=markers[0], label=v)
    for i, v in enumerate(true_cluster_names)
])

plt.show()
