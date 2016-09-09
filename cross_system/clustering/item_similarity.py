import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.clusterings import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

n_clusters = 5
answers = None
items = None
true_cluster_names = None
gt_column = None
# MATMAT
if True:
    answers = pd.read_pickle('../../data/matmat/2016-06-27/answers.pd')
    answers = answers.groupby(['student', 'item']).first().reset_index()
    items = pd.read_csv('../../data/matmat/2016-06-27/items.csv', index_col='id')
    items = items[items['visualization'] != 'pairing']
    items = items[items['skill_lvl_1'] == 2]
    gt_column = 'visualization'

    true_cluster_names = list(items['visualization'].unique())
    answers = answers[answers['item'].isin(items.index)]

kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000)

# modificator = BinaryResponse()
# modificator = TimeLimitResponseModificator([(5, 0.5)])
modificator = LinearDrop(14)
answers = modificator.modify(answers)

projection = pca


plt.figure(figsize=(16, 24))
similarities, similarities_names = [], ['ground_truth']
for g in [None, similarity_pearson]:
    for f in [similarity_pearson, similarity_cosine, similarity_kappa, similarity_euclidean]:
        if f is None:
            continue
        if g is not None:
            similarities.append(lambda x, g=g, f=f: g(f(x)))
            similarities_names.append("{} -> {}".format(f.__name__.replace('similarity_', ''), g.__name__.replace('similarity_', '')))
        else:
            similarities.append(lambda x, f=f: f(x))
            similarities_names.append(f.__name__.replace('similarity_', ''))


clusters = []
for i, similarity in enumerate(similarities):
    print(similarities_names[i])
    X = similarity(answers)
    labels = kmeans.fit_predict(X)
    clusters.append(labels)

    items_ids = X.index
    (xs, ys), _ = projection(X, clusters=n_clusters)
    ground_truth = [true_cluster_names.index(items.get_value(item, gt_column)) for item in items_ids]

    plt.subplot(2, len(similarities) / 2 + 1, i + 1)
    plt.title(similarities_names[i+1])
    plot_clustering(
        items_ids, xs, ys,
        labels=labels,
        texts=[items.get_value(item, 'question') for item in items_ids],
        shapes=ground_truth,
    )

plt.legend(handles=[
    mlines.Line2D([], [], color='black', linewidth=0, marker=markers[i], label=v)
    for i, v in enumerate(true_cluster_names)
    ])


plt.subplot(2, len(similarities) / 2 + 1, len(similarities) + 1)
rands = []
for c1 in [ground_truth] + clusters:
    l = []
    for c2 in [ground_truth] + clusters:
        l.append(rand_index(c1, c2))
    rands.append(l)

sns.heatmap(rands, xticklabels=similarities_names, yticklabels=similarities_names, annot=True)
sns.clustermap(rands, xticklabels=similarities_names, yticklabels=similarities_names, annot=True)


# plt.savefig('results/tmp/matmat-{}-x.png'.format(modificator))
plt.show()

