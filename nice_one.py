from collections import defaultdict

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

from models.eloConcepts import EloConcepts
from models.eloPriorCurrent import EloPriorCurrentModel
from utils import data as d
from utils.data import LinearDrop, items_in_concept
from utils.runner import Runner, get_hash
from utils.utils import Cache, enumerate_df


@Cache(cache_name_var='dataset')
def enum(dataset=None):
    data = pd.read_pickle('data/mathgarden/{}.pd'.format(dataset))
    data = data.groupby('student').apply(enumerate_df)
    return data

length = 140
lim = 100
pal = sns.color_palette()

curves = {}

datasets = ['addition', 'subtraction', 'multiplication']
for dataset in datasets:
    data = enum(dataset=dataset)
    raw = data.groupby('enum')['correct'].mean()[:length]
    curves[dataset] = {
        'raw': raw,
        'learners': data.groupby('enum').apply(len)[:length],
        'diff': raw - raw.shift(1),
    }
    students = data.groupby('student').apply(len)
    long_students = students[students > lim]
    curves[dataset]['student_selection'] = data[data['student'].isin(long_students)].groupby('enum')['correct'].mean()[:lim]


plt.figure(figsize=(10, 15))

plt.suptitle('Learning')

plt.subplot(3, 1, 1)
for i, (dataset, data) in enumerate(curves.items()):
    plt.plot(data['raw'], alpha=0.2, color=pal[i], label='')
    plt.plot(data['raw'].rolling(10, center='true').mean(), color=pal[i], label=dataset)
    # plt.plot(data['student_selection'].rolling(1, center='true').mean(), color=pal[i], label=dataset)
plt.legend()

plt.subplot(3, 1, 2)
for i, (dataset, data) in enumerate(curves.items()):
    plt.plot(data['diff'], alpha=0.2, color=pal[i])
    plt.plot(data['diff'].rolling(10, center='true').mean(), color=pal[i], label=dataset)
    plt.ylim(-0.01, 0.01)

plt.subplot(3, 1, 3)
for i, (dataset, data) in enumerate(curves.items()):
    plt.plot(data['learners'])

plt.show()