import seaborn as sns
from sklearn.cluster import KMeans

from cross_system.clustering.clusterings import *
from cross_system.clustering.similarity import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

answers = pd.read_pickle('../../../data/matmat/2016-06-27/answers.pd')
answers = answers.groupby(['student', 'item']).first().reset_index()
items = pd.read_csv('../../../data/matmat/2016-06-27/items.csv', index_col='id')
items = items[items['visualization'] != 'pairing']
n_clusters = 3

# print(items['skill_lvl_1'].unique()) # [  2  26 151 209 367]
items = items[items['skill_lvl_1'] == 2]
# items = items[items['skill_lvl_2'] == 27]
# items = items[items['skill_lvl_2'] == 210]
# items = items[items['skill_lvl_2'] == 152]
visualizations = list(items['visualization'].unique())
answers = answers[answers['item'].isin(items.index)]
kmeans = KMeans(n_clusters=n_clusters)

# modificator = BinaryResponse()
# modificator = TimeLimitResponseModificator([(5, 0.5)])
modificator =  LinearDrop(14)
answers = modificator.modify(answers)

X = similarity_pearson(similarity_pearson(answers))
# clusters = kmeans.fit_predict(X)


plt.figure(figsize=(15, 15))
for i, method in enumerate([tsne, pca, spectral_clustering]):
    items_ids = X.index
    (xs, ys), labels = method(X, clusters=n_clusters)

    plt.subplot(2, 2, i + 1)
    plt.title(method.__name__)
    plot_clustering(
        items_ids, xs, ys,
        labels=clusters,
        texts=[items.get_value(item, 'question') for item in items_ids],
        shapes=[visualizations.index(items.get_value(item, 'visualization')) for item in items_ids],
    )

plt.subplot(2, 2, 4)
plt.title(method.__name__ + " - own clustering")
plot_clustering(
    items_ids, xs, ys,
    labels=labels,
    texts=[items.get_value(item, 'question') for item in items_ids],
    shapes=[visualizations.index(items.get_value(item, 'visualization')) for item in items_ids],
)

plt.legend(handles=[
    mlines.Line2D([], [], color='black', linewidth=0, marker=markers[i], label=v)
    for i, v in enumerate(visualizations)
])

plt.savefig('../results/tmp/matmat-{}-x.png'.format(modificator))
plt.show()

