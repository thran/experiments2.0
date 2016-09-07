from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pylab as plt
import seaborn as sns
from algorithms.spectralclustering import SpectralClusterer
from cross_system.clustering.clusterings import *

colors = "rgbyk"



answers_cs = pd.read_pickle('../../data/umimecesky/2016-05-18/answers.pd')
answers_cs = answers_cs.groupby(['student', 'item']).first().reset_index()
items_cs = pd.read_pickle('../../data/umimecesky/2016-05-18/items.pd')


def embeddings(answers, corr_method='corr'):
    if corr_method == 'kappa':
        corr = kappa(answers)
    elif corr_method == 'corrcorr':
        vectorization_double_pearson(answers)
    else:
        vectorization_pearson(answers)
    items = corr.index
    distances = pd.DataFrame()

    for i, method in enumerate([spectral_clustering, tsne, pca, isomap, mds]):
        (xs, ys), labels = method(corr)

        distances[method.__name__] = pdist(np.array([xs, ys]).T)

        plt.subplot(2, 3, i + 1)
        plt.title(method.__name__)
        plot_clustering(
            items, xs, ys,
            [items_cs.get_value(item, 'correct_variant') for item in items],
            [items_cs.get_value(item, 'solved') for item in items]
        )
    plt.subplot(2, 3, 6)
    sns.heatmap(distances.corr(), vmin=0.5)

answers = answers_cs[answers_cs['item'].isin(items_cs.loc[items_cs['concept'] == 2].index)]
# embeddings(answers, corr_method='kappa')
# embeddings(answers, corr_method='corr')
embeddings(answers, corr_method='corrcorr')
plt.show()