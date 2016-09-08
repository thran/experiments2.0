import seaborn as sns
from cross_system.clustering.clusterings import *
import pandas as pd
import os


answers = pd.read_pickle('../../../data/matmat/2016-06-27/answers.pd')
answers = answers.groupby(['student', 'item']).first().reset_index()
items = pd.read_csv('../../../data/matmat/2016-06-27/items.csv', index_col='id')
items = items[items['visualization'] != 'pairing']

# print(items['skill_lvl_1'].unique()) # [  2  26 151 209 367]
# items = items[items['skill_lvl_1'] == 2]
items = items[items['skill_lvl_2'] == 27]
# items = items[items['skill_lvl_2'] == 210]
# items = items[items['skill_lvl_2'] == 152]
visualizations = list(items['visualization'].unique())
answers = answers[answers['item'].isin(items.index)]

X = vectorization_double_pearson(answers)

for i, method in enumerate([tsne, tsne, pca, isomap, mds]):
    items_ids = X.index
    (xs, ys), labels = method(X)

    plt.subplot(2, 3, i + 1)
    plt.title(method.__name__)
    plot_clustering(
        items_ids, xs, ys,
        [visualizations.index(items.get_value(item, 'visualization')) for item in items_ids],
        [items.get_value(item, 'question') for item in items_ids]
    )

plt.show()


