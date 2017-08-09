import seaborn as sns
from sklearn.cluster import KMeans

from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

answers = pd.read_pickle('../../data/matmat/2016-06-27/answers.pd')
answers = answers.groupby(['student', 'item']).first().reset_index()
items = pd.read_csv('../../data/matmat/2016-06-27/items.csv', index_col='id')
items = items[items['visualization'] != 'pairing']

# print(items['skill_lvl_1'].unique()) # [  2  26 151 209 367]
# items = items[items['skill_lvl_1'] == 2]
# items = items[items['skill_lvl_2'] == 27  ]
# items = items[items['skill_lvl_2'] == 210]
items = items[items['skill_lvl_2'] == 152]
visualizations = list(items['visualization'].unique())
answers = answers[answers['item'].isin(items.index)]
print(visualizations)



embeddings = [pca, mds, tsne, spectral]
for i, embedding in enumerate(embeddings):
    X = similarity_pearson(similarity_pearson(answers))
    items_ids = X.index
    xs, ys = embedding(X, euclid=True)
    plt.subplot(1, len(embeddings), i + 1)
    plt.title(embedding.__name__)
    plot_clustering(
        items_ids, xs, ys,
        # labels=clusters,
        texts=[items.get_value(item, 'question') for item in items_ids],
        labels=[visualizations.index(items.get_value(item, 'visualization')) for item in items_ids],
    )

plt.show()