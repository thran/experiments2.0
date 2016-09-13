import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.clusterings import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

# data_set, n_clusters  = 'matmat-numbers', 3
# data_set, n_clusters  = 'matmat-all', 4
data_set, n_clusters  = 'simulated-s100-c5-i20', 5
# data_set, n_clusters  = 'simulated-s250-c2-i20', 2
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-multiplication', 1
# data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000)

modificator = BinaryResponse()
# modificator = TimeLimitResponseModificator([(5, 0.5)])
# modificator = LinearDrop(14)
answers = modificator.modify(answers)

projection = tsne


plt.figure(figsize=(16, 24))
plt.suptitle('{} - {}'.format(data_set, modificator))
similarities, similarities_names = [], []

for f in [similarity_yulesQ, similarity_pearson, similarity_kappa, similarity_euclidean, similarity_cosine]:
# for f in [similarity_pearson, similarity_euclidean, similarity_cosine]:
    for g in [None, similarity_pearson]:
        if f is None:
            continue
        if g is not None:
            similarities.append(lambda x, g=g, f=f: g(f(x, cache=data_set + str(modificator))))
            similarities_names.append("{} -> {}".format(f.__name__.replace('similarity_', ''), g.__name__.replace('similarity_', '')))
        else:
            similarities.append(lambda x, f=f: f(x, cache=data_set + str(modificator)))
            similarities_names.append(f.__name__.replace('similarity_', ''))


if False:
    clusters = []
    for i, similarity in enumerate(similarities):
        print(similarities_names[i])
        X = similarity(answers)
        labels = kmeans.fit_predict(X)
        clusters.append(labels)

        items_ids = X.index
        (xs, ys), _ = projection(X, clusters=n_clusters)
        ground_truth = [true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids]

        plt.subplot(2, len(similarities) / 2 + 1, i + 1)
        plt.title(similarities_names[i])
        plot_clustering(
            items_ids, xs, ys,
            labels=labels,
            texts=[items.get_value(item, 'name') for item in items_ids],
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

    sns.heatmap(rands, xticklabels=['truth'] + similarities_names, yticklabels=['truth'] + similarities_names, annot=True)
    plt.title(data_set)
    sns.clustermap(rands, xticklabels=['truth'] + similarities_names, yticklabels=['truth'] + similarities_names, annot=True)


if True:
    for i, (similarity, similarities_name) in enumerate(zip(similarities, similarities_names)):
        print(similarities_name)
        X = similarity(answers)
        ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
        same, different = [], []
        for concept1 in set(ground_truth):
            for concept2 in set(ground_truth):
                values = list(X.loc[ground_truth == concept1, ground_truth == concept2].values.flatten())
                if concept1 == concept2:
                    same += values
                elif concept1 > concept2:
                    different += values


        # plt.subplot(len(similarities) // 2, 4, i + 1 + i // 2 * 2)
        plt.subplot(len(similarities) // 2, 2, i + 1)
        plt.title(similarities_name)
        sns.distplot(same)
        if len(different):
            sns.distplot(different)
        if not similarities_name.endswith('euclidean'):
            plt.xlim([-1,1])

        if i % 2 == 1 and False:
            predict = kmeans.fit_predict(X)
            plt.subplot(len(similarities) // 2, 4, i + 2 + i // 2 * 2)
            (xs, ys), _ = tsne(X, clusters=n_clusters)
            plot_clustering(
                X.index, xs, ys,
                labels=predict,
                # texts=[items.get_value(item, 'name') for item in X.index],
                texts=None,
                shapes=ground_truth,
            )

            plt.subplot(len(similarities) // 2, 4, i + 3 + i // 2 * 2)
            (xs, ys), _ = pca(X, clusters=n_clusters)
            plot_clustering(
                X.index, xs, ys,
                labels=predict,
                texts=[items.get_value(item, 'name') for item in X.index],
                shapes=ground_truth,
            )

    plt.legend(handles=[
        mlines.Line2D([], [], color='black', linewidth=0, marker=markers[i], label=v)
        for i, v in enumerate(true_cluster_names)
    ])


if False:
    plt.figure(figsize=(8, 15))
    plt.suptitle('{}'.format(data_set))
    similarity = similarity_double_pearson

    embeddings = [pca, isomap, mds, spectral_clustering, tsne]
    for i, embedding in enumerate(embeddings):
        X = similarity(answers)
        ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
        (xs, ys), _ = embedding(X)
        plt.subplot(2, len(embeddings) // 2 + 1, i + 1)
        plt.title(embedding.__name__)
        plot_clustering(
            X.index, xs, ys,
            labels=ground_truth,
            # texts=[items.get_value(item, 'name') for item in X.index],
            shapes=None,
        )


    # plt.savefig('results/tmp/matmat-{}-x.png'.format(modificator))
    plt.show()



# plt.savefig('results/tmp/matmat-{}-x.png'.format(modificator))
plt.show()

