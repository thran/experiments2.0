from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from mastery_learning.mastery_metrics import *
from mastery_learning.mastery_utils import *
from models.eloConcepts import EloConcepts
from utils import data as d
from utils.data import LinearDrop

data = d.Data("../data/matmat/2016-11-28/answers.pd")
data.trim_times()
concepts = data.get_concepts()

data_time = d.Data("../data/matmat/2016-11-28/answers.pd", response_modification=LinearDrop(14))
data_time.trim_times()


def model(): return EloConcepts(concepts=concepts, separate=True)
add_skills(data, model())
add_difficulties(data, model())
add_skills(data_time, model(), 'skill_time', data)
add_difficulties(data_time, model(), 'difficulty_time', data)
data.get_dataframe_test()['correct_time'] = data_time.get_dataframe_test()['correct']

concept = 'numbers'
# concept = 'addition'
# concept = 'multiplication'
low_limit = 0

names = {
    model_skills: 'S',
    model_skills_time: 'S + T',
    exponential_average: 'EWA',
    exponential_average_difficulty: 'EWA + D',
    exponential_average_time: 'EWA + T',
    exponential_average_difficulty_time: 'EWA + D + T'
}


if 0:
    types = [exponential_average, model_skills, exponential_average_difficulty, exponential_average_time, exponential_average_difficulty_time, model_skills_time]
    df = pd.DataFrame()
    for type in types:
        cache_name = '{}-{}-{}'.format(data, concept, type.__name__)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        df[names[type]] = get_order(curves)

    print(df.corr(method='spearman'))
    sns.heatmap(df.corr(method='spearman'))


#### mastery metrics
if 1:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for type in [exponential_average, model_skills, exponential_average_difficulty, exponential_average_time, exponential_average_difficulty_time, model_skills_time]:
        cache_name = '{}-{}-{}'.format(data, concept, type.__name__)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        df = thresholds(curves, cache=cache_name)
        ax1.plot(df.columns[low_limit:], [df[point].mean() for point in df.columns[low_limit:]], label=names[type])
        ax2.plot(df.columns[low_limit:], [sum(~df[point].isnull()) for point in df.columns[low_limit:]], label=names[type])
    ax1.set_ylabel('avg attempts to mastery')
    ax2.set_ylabel('learners')


#### alpha
if 0:
    for alpha in [0.7, 0.8, 0.9, 0.95]:
        type = partial(exponential_average, exp=alpha)
        cache_name = '{}-{}-exp{}'.format(data, concept, alpha)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        df = thresholds(curves, cache=cache_name)

        plt.plot(df.columns[low_limit:], [df[point].mean() for point in df.columns[low_limit:]], label=alpha)

plt.xlabel('threshold')
plt.legend()
plt.title(concept)

plt.show()
