import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from cross_system.clustering.similarity import similarity_pearson
from utils.data import LinearDrop

for i, dataset in enumerate(['matmat-numbers', 'matmat-addition', 'matmat-multiplication', 'math_garden-addition']):
    print(dataset)

    answers = pd.read_pickle('../cross_system/clustering/data/{}-answers.pd'.format(dataset))
    items = pd.read_pickle('../cross_system/clustering/data/{}-items.pd'.format(dataset))
    sims = pd.DataFrame()

    sims['without_time'] = similarity_pearson(similarity_pearson(answers)).replace(1, 0).values.flatten()

    modificator = LinearDrop(2 * answers['response_time'].median())
    answers = modificator.modify(answers)
    sims['with_time'] = similarity_pearson(similarity_pearson(answers)).replace(1, 0).values.flatten()

    sns.jointplot('without_time', 'with_time', data=sims.loc[:1000, :], size=4, )

plt.show()