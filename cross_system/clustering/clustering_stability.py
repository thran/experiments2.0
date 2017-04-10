from collections import defaultdict

import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.clusterings import *
from cross_system.clustering.projection import *
from cross_system.clustering.similarity import *
import pandas as pd
import os
import matplotlib.lines as mlines
import matplotlib.pylab as plt
from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse

# similarity_setting = lambda x: similarity_pearson(x), 'pearson'
# similarity_setting = similarity_yulesQ, 'yuleQ'
# similarity_setting = similarity_pearson, 'pearson'
# similarity_setting = lambda x: similarity_yulesQ(x), 'yuleQ'
# similarity_setting = lambda x: similarity_pearson(similarity_pearson(x)), 'pearson -> pearson'
similarity_setting = similarity_jaccard, 'Jaccard'
runs = 10
results = []

for data_set in [
        # 'simulated-s100-c5-i20',
        # 'simulated-s250-c2-i20',
        'matmat-numbers',
        'matmat-addition',
        'math_garden-addition',
        'math_garden-multiplication',
        'cestina-B',
        'cestina-konc-prid',
    ]:
    answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
    items = pd.read_pickle('data/{}-items.pd'.format(data_set))
    true_cluster_names = list(items['concept'].unique())
    students = pd.Series(answers['student'].unique())

    # print(data_set, len(students), len(items), len(answers))
    # continue

    for frac in list(np.arange(0.02, 0.2, 0.02)) + list(np.arange(0.2, 1.1, 0.1)):
        for run in range(runs):
            S = students.sample(frac=frac)
            S1 = S[:len(S) // 2]
            S2 = S[len(S) // 2:]
            A1 = answers[answers['student'].isin(S1)]
            A2 = answers[answers['student'].isin(S2)]
            print(data_set, frac, len(A1), len(A2))
            similarity, similarity_name = similarity_setting
            X1 = similarity(A1)
            X2 = similarity(A2)
            if len(X1.index) != len(X2.index):
                continue

            p, _ = pearsonr(X1.replace(1, 0).as_matrix().flatten(), X2.replace(1, 0).as_matrix().flatten())

            results.append([frac, p, run, data_set])

results = pd.DataFrame(results, columns=['frac', 'correlation', 'run', 'data_set'])
print(results)

plt.figure(figsize=(16, 24))
plt.title(similarity_name)
# sns.pointplot(data=results, x='frac', y='rand_index', hue='clustering')
sns.tsplot(data=results, time='frac', value='correlation', unit='run', condition='data_set')

plt.show()
