import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index
from scipy.stats import pearsonr
from cross_system.clustering.similarity import *
from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse, \
    MathGardenResponseModificator

# data_set, n_clusters  = 'matmat-numbers', 3
# data_set, n_clusters  = 'matmat-multiplication', 1
# data_set, n_clusters  = 'matmat-addition', 1
# data_set, n_clusters  = 'matmat-all', 4
# data_set, n_clusters  = 'math_garden-multiplication', 1
data_set, n_clusters  = 'math_garden-addition', 1
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-all2', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

median = np.round(np.median(answers['response_time']), 1)
# modificators = [BinaryResponse(), TimeLimitResponseModificator([(median, 0.5)]), MathGardenResponseModificator(2 * median), LinearDrop(2 * median)]
modificators = [BinaryResponse(), LinearDrop(2 * median)]

similarity, euclid = similarity_pearson, True
projection = tsne
clustering = kmeans

# answers = answers.loc[:67000, :]
print(len(answers))

if False:
    plt.figure(figsize=(10, 5))
    # students = answers['student'].unique()
    # students = students[: len(students) // 2]
    # answers = answers[answers['student'].isin(students)]
    for i, modificator in enumerate(modificators):
        print(modificator, len(answers))
        modified_answers = modificator.modify(answers.copy())
        X = similarity(modified_answers)
        xs, ys = projection(X, euclid=euclid)
        ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])

        plt.subplot(1, len(modificators), i + 1)
        plt.title(str(modificator))

        for x, y, item, visualization in zip(xs, ys, X.index, ground_truth):
            value = items.get_value(item, 'name')
            plt.plot(x, y, markers[visualization], color=colors[visualization], alpha=(int(value) + 5) / 25)
            plt.text(x, y, value)



    plt.legend(handles=[
        mlines.Line2D([], [], color=colors[i], linewidth=0, marker=markers[i], label=v)
        for i, v in enumerate(true_cluster_names)
        ])

if False:
    # answers = answers.sample(n=20000)
    print(len(answers))
    data = pd.DataFrame()
    for modificator in modificators:
        print(modificator)
        modified_answers = modificator.modify(answers.copy())
        # print(modified_answers)
        X = similarity(modified_answers)
        data[str(modificator)] = X.as_matrix().flatten()

    data = data.replace([1],0)
    print(data.corr().as_matrix().round(2))
    # print(data.corr(method='spearman'))

    g = sns.pairplot(data, diag_kind="kde")
    g.map_lower(sns.kdeplot, cmap="Blues_d")

if True:
    results = []
    truths = {}
    s = None
    for modificator in modificators:
        modified_answers = modificator.modify(answers.copy())
        X = similarity(modified_answers)
        c = X.replace(1, 0).as_matrix().flatten()
        truths[str(modificator)] = c
        s = c if s is None else s + c
    truths['avg'] = s / len(truths)

    for run in range(10):
        for frac in np.concatenate([np.arange(0.02, .41, 0.02), np.arange(0.6, 1.1, 0.2)]):
            sampled_answers = answers.sample(frac=frac)
            for modificator in modificators:
                print(run, frac, modificator)
                modified_answers = modificator.modify(sampled_answers.copy())
                X = similarity(modified_answers)
                c = X.replace(1, 0).as_matrix().flatten()
                for final_modificator, truth in truths.items():
                    p, _ = pearsonr(truth, c)
                    results.append([
                        str(modificator),
                        str(final_modificator),
                        frac,
                        p,
                        run
                    ])

    results = pd.DataFrame(results, columns=['time_use', 'target_time_use', 'sample', 'pearson', 'run'])
    for modificator in modificators + ['avg']:
        plt.figure()
        plt.title(str(modificator))
        # sns.pointplot(data=results[results['target_time_use'] == str(modificator)], x='sample', y='pearson', hue='time_use')
        sns.tsplot(data=results[results['target_time_use'] == str(modificator)], time='sample', value='pearson', unit='run', condition='time_use')


plt.show()
