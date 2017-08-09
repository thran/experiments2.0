import seaborn as sns
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.similarity import *
from cross_system.clustering.projection import *
from cross_system.clustering.clusterings import *
import matplotlib.pylab as plt
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

# data_set, n_clusters  = 'matmat-numbers', 3
# data_set, n_clusters  = 'matmat-all', 4
# data_set, n_clusters  = 'simulated-s100-c5-i20', 2
# data_set, n_clusters  = 'simulated-s100-c5-i200', 5
# data_set, n_clusters  = 'simulated-s50-c5-i100', 5
# data_set, n_clusters  = 'simulated-s250-c2-i20', 2
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-multiplication', 3
data_set, n_clusters = 'cestina-B', 2
# data_set, n_clusters = 'cestina-L', 2
# data_set, n_clusters = 'cestina-Z', 2
# data_set, n_clusters = 'cestina-konc-prid', 7
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

print(data_set, len(answers), len(items))

modificator = BinaryResponse()
# modificator = TimeLimitResponseModificator([(5, 0.5)])
# modificator = LinearDrop(14)
answers = modificator.modify(answers)

projection = tsne



plt.figure(figsize=(15, 10))
plt.suptitle('{} - {}'.format(data_set, modificator))
similarities, similarities_names = [], []

# for f in [similarity_yulesQ, similarity_pearson, similarity_kappa, similarity_euclidean, similarity_cosine]:
for f in [similarity_pearson, similarity_yulesQ]:
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
    euclid = True
    clusters = []
    for i, similarity in enumerate(similarities):
        print(similarities_names[i])
        X = similarity(answers)
        labels = kmeans(X, concepts=n_clusters, euclid=euclid)
        clusters.append(labels)

        items_ids = X.index
        xs, ys = projection(X, euclid=euclid)
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
    # sns.clustermap(rands, xticklabels=['truth'] + similarities_names, yticklabels=['truth'] + similarities_names, annot=True)


if False:
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
        if not similarities_name.endswith('euclidean'):
            plt.xlim([-1,1])
            sns.distplot(same, label='inner')
            if len(different):
                sns.distplot(different, label='outer')
        else:
            plt.xlim([-max(different), 0])
            sns.distplot(-np.array(same))
            if len(different):
                sns.distplot(-np.array(different))
        plt.legend()

        if i % 2 == 1 and False:
            plt.subplot(len(similarities) // 2, 4, i + 2 + i // 2 * 2)
            (xs, ys) = tsne(X)
            plot_clustering(
                X.index, xs, ys,
                # labels=predict,
                # texts=[items.get_vais lue(item, 'name') for item in X.index],
                texts=None,
                shapes=ground_truth,
            )

            plt.subplot(len(similarities) // 2, 4, i + 3 + i // 2 * 2)
            (xs, ys) = pca(X)
            plot_clustering(
                X.index, xs, ys,
                # labels=predict,
                texts=[items.get_value(item, 'name') for item in X.index],
                shapes=ground_truth,
            )

if False:
    similarity, euclid = similarity_pearson, True

    embeddings = [pca, mds, tsne, spectral]
    for i, embedding in enumerate(embeddings):

        ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
        xs, ys = embedding(X, euclid=euclid)
        plt.subplot(1, len(embeddings), i + 1)
        plt.title(embedding.__name__)
        plot_clustering(
            X.index, xs, ys,
            labels=ground_truth,
            texts=[items.get_value(item, 'name') for item in X.index],
            shapes=None,
        )

    if False:
        plt.legend(handles=[
            mlines.Line2D([], [], color=colors[i], linewidth=0, marker=markers[0], label=v)
            for i, v in enumerate(true_cluster_names)
        ])
    plt.show()


if True:
    X = similarity_pearson(similarity_pearson(answers))
    xs, ys = tsne(X, euclid=False)
    ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
    for clustering in [kmeans, hierarchical, spectral_clustering]:
        print(rand_index(ground_truth, clustering(X)))

    plot_clustering(
        X.index, xs, ys,
        labels=ground_truth,
        # texts=[items.get_value(item, 'name') for item in X.index],
        shapes=None,
    )

# plt.savefig('results/tmp/matmat-{}-x.png'.format(modificator))
plt.show()

