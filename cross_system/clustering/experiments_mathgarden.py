import seaborn as sns
from cross_system.clustering.clusterings import *
import pandas as pd
import os
import matplotlib.lines as mlines


if False:
    answers = pd.read_pickle('../../data/mathgarden/multiplication.pd')
    answers = answers.rename(columns={'user_id': 'student', 'item_id': 'item', 'correct_answered': 'correct'})
    answers = answers.groupby(['student', 'item']).first().reset_index()
    print(answers['item'].unique())

    X = similarity_double_pearson(answers)
    items_ids = X.index
    (xs, ys), labels = mds(X)
    plot_clustering(
        items_ids, xs, ys,
        [0 for item in items_ids],
        [item for item in items_ids]
    )

if True:
    data_set, n_clusters  = 'math_garden-all', 3
    answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    items = pd.read_pickle('data/{}-items.pd'.format(data_set))
    true_cluster_names = list(items['concept'].unique())
    answers['response_time_log'] = np.log(answers['response_time'])

    print(answers['student'])

    success_rate = answers.groupby('item')['correct'].mean()

    response_time = answers.groupby('item')['response_time_log'].mean()
    plot_clustering(
        items.index, success_rate[items.index], response_time[items.index],
        labels=[true_cluster_names.index(items.get_value(item, 'concept')) for item in items.index],
        texts=None,
        shapes=[true_cluster_names.index(items.get_value(item, 'concept')) for item in items.index],
    )

    plt.title('MathGarden dataset')
    plt.xlabel('success rate')
    plt.ylabel('median of response time')

    plt.legend(handles=[
        mlines.Line2D([], [], color='black', linewidth=0, marker=markers[i], label=v)
        for i, v in enumerate(true_cluster_names)
    ])

plt.show()


