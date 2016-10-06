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
    data_set, n_clusters = 'cestina-konc-prid', 7
    answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    items = pd.read_pickle('data/{}-items.pd'.format(data_set))
    true_cluster_names = list(items['concept'].unique())


    projection = tsne
    similarity, euclid = similarity_pearson, True


    X = similarity(answers)
    X2 = pd.read_csv('doplnovacka-konc-prid', header=None)
    print(X2)
    xs, ys = projection(X, euclid=euclid)
    # model = TSNE(n_components=2, learning_rate = 100, n_iter=200000)
    # xs, ys = model.fit_transform(X2).T


    items_ids = X.index
    ground_truth = np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in items_ids])

    plot_clustering(
        items_ids, xs, ys,
        labels=ground_truth,
        # texts=[items.get_value(item, 'name') for item in items_ids],
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


