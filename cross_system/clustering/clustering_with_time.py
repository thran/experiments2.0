import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as rand_index

from cross_system.clustering.clusterings import *
import pandas as pd
import os
import matplotlib.lines as mlines

from utils.data import TimeLimitResponseModificator, LinearDrop, BinaryResponse


def sample(answers, n=None, ratio=1):
    if n is not None:
        return answers.sample(n=n)
    return answers.sample(n=int(ratio * len(answers)))

data_set, n_clusters  = 'matmat-all', 4
# data_set, n_clusters  = 'math_garden-all', 3
# data_set, n_clusters  = 'math_garden-all2', 2
answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))
true_cluster_names = list(items['concept'].unique())

kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000)

modificators = [BinaryResponse(), TimeLimitResponseModificator([(5, 0.5)]), LinearDrop(14)]
# modificator = BinaryResponse()
# modificator = TimeLimitResponseModificator([(5, 0.5)])
# modificator = LinearDrop(14)


plt.figure(figsize=(16, 24))
plt.suptitle('{}'.format(data_set))
similarity = similarity_double_pearson
sample_sizes = range(5 * 10**4, 15 * 10**4 + 1, 2 *10**4)
print(len(answers))
runs = 2

data = []
for modificator in modificators:
    print(modificator)
    modified_answers = modificator.modify(answers)
    for sample_size in sample_sizes:
        print(str(sample_size))
        for i in range(runs):
            X = similarity(sample(modified_answers, n=sample_size))
            ground_truth =np.array([true_cluster_names.index(items.get_value(item, 'concept')) for item in X.index])
            labels = kmeans.fit_predict(X)
            rand = rand_index(ground_truth, labels)
            print(''  + str(round(rand, 2)))
            data.append([sample_size, rand,modificator])

df = pd.DataFrame(data, columns=['sample_size', 'rand_index', 'time_use'])
sns.pointplot(data=df, x='sample_size', y='rand_index', hue='time_use')

# plt.savefig('results/tmp/matmat-{}-x.png'.format(modificator))
plt.show()

