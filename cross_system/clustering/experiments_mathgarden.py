import seaborn as sns
from cross_system.clustering.clusterings import *
import pandas as pd
import os


answers = pd.read_pickle('../../data/mathgarden/multiplication.pd')
answers = answers.rename(columns={'user_id': 'student', 'item_id': 'item', 'correct_answered': 'correct'})
answers = answers.groupby(['student', 'item']).first().reset_index()
print(answers['item'].unique())

X = vectorization_double_pearson(answers)
items_ids = X.index
(xs, ys), labels = mds(X)
plot_clustering(
    items_ids, xs, ys,
    [0 for item in items_ids],
    [item for item in items_ids]
)

plt.show()


