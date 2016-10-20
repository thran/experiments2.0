import math
import random
import seaborn as sns
import numpy as np
import pandas as pd
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
from sklearn.metrics import adjusted_rand_score as rand_index


if True:
    # data_set, n_clusters = 'cestina-konc-prid', 7
    # data_set, n_clusters  = 'math_garden-multiplication', 3
    # data_set, n_clusters  = 'math_garden-addition', 3
    # data_set, n_clusters  = 'math_garden-subtraction', 3
    # data_set, n_clusters  = 'math_garden-all', 3
    answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    items = pd.read_pickle('data/{}-items.pd'.format(data_set))
    true_cluster_names = list(items['concept'].unique())

    print(len(answers), len(items))

    projection = tsne
    similarity, euclid = similarity_pearson, True


    X = similarity(answers)
    xs, ys = projection(X, euclid=euclid, perplexity=10)

    items_ids = X.index
    ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

    plot_clustering(
        items_ids, xs, ys,
        labels=ground_truth,
        texts=[items.get_value(item, 'name') for item in items_ids],
        shapes=ground_truth,
    )


if False:
    df = pd.read_csv('tsne-prid-jmena-data.csv', sep=';')
    print(df)
    shapes = list(df['shape'].unique())
    plot_clustering(
        df.index, df['x'], df['y'],
        labels=df['concept'],
        # texts=df['word'],
        shapes=[shapes.index(s) for s in df['shape'].values],
    )

plt.show()


