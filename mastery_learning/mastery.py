from collections import defaultdict
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

# data = d.Data("../data/matmat/2016-11-28/answers.pd")
data = d.Data("../data/matmat/2016-11-28/answers.pd", filter=(100, 11))
data.trim_times()
concepts = data.get_concepts()

# data_time = d.Data("../data/matmat/2016-11-28/answers.pd", response_modification=LinearDrop(14))
data_time = d.Data("../data/matmat/2016-11-28/answers.pd", response_modification=LinearDrop(14), filter=(100, 11))
data_time.trim_times()


# print(len(data.get_students()))
# print(len(data.get_items()))
# print(len(data.get_dataframe_all()))

def model(): return EloConcepts(concepts=concepts, separate=True)
add_skills(data, model())
add_difficulties(data, model())
add_skills(data_time, model(), 'skill_time', data)
add_difficulties(data_time, model(), 'difficulty_time', data)
data.get_dataframe_test()['correct_time'] = data_time.get_dataframe_test()['correct']

# concept = 'numbers'
# concept = 'addition'
concept = 'multiplication'
# concept = 'division'
# concept = 'subtraction'
low_limit = 0

names = {
    model_skills: 'M',
    model_skills_time: 'M + T',
    exponential_average: 'EWA',
    exponential_average_difficulty: 'EWA + D',
    exponential_average_time: 'EWA + T',
    exponential_average_difficulty_time: 'EWA + D + T'
}


if 1:
    n = 5
    # thresholds, type = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98], exponential_average
    thresholds, type = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], exponential_average_time
    export = {}
    for concept in ['numbers', 'addition', 'multiplication', 'division', 'subtraction']:
        cache_name = '{}-{}-{}'.format(data, concept, type.__name__)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        pointsx = []
        pointsy = []
        for threshold in thresholds:
            means = []
            efforts = []
            for curve in curves:
                for i, x in enumerate(curve):
                    if x >= threshold:
                        sub = []
                        for y in curve[i+1:i+1+n]:
                            sub.append(1 if y > x else 0)
                            x = y
                        if len(sub) == n:
                            means.append(np.mean(sub))
                            efforts.append(i + 1)
                        break
            print(concept, threshold, np.mean(means))
            pointsy.append(np.mean(means))
            pointsx.append(np.mean(efforts))
        export[concept] = {
            'efforts': pointsx,
            'scores': pointsy,
            'thresholds': thresholds,
        }
        plt.plot(pointsx, pointsy, label=concept, marker='*', markersize=10)
    plt.title('thresholds: {}'.format(thresholds))
    print(export)


if 0:
    points = list(np.arange(0.01, 0.3, 0.01)) + list(np.arange(0.3, 1, 0.1))
    for type1, type2 in [
        (model_skills_time, exponential_average_time),
        (exponential_average, exponential_average_time),
        (exponential_average, model_skills_time),
        (exponential_average, model_skills),
        (exponential_average_time, model_skills),
        (model_skills, model_skills_time),
    ]:
        jaccard = []
        for percent in points:
            mastereds = {}
            for type in [type1, type2]:
                cache_name = '{}-{}-{}'.format(data, concept, type.__name__)
                curves = get_mastery_curves(data, concept, type, cache=cache_name)
                df = thresholds(curves, cache=cache_name)
                student_count = len(curves)
                trashold = ((~df.isnull()).sum(axis=0) < student_count * percent).argmax()
                mastered = set(df.index[~df[trashold].isnull()])
                mastereds[names[type]] = mastered
            jaccard.append(len(mastereds[names[type1]] & mastereds[names[type2]]) / len(mastereds[names[type1]] | mastereds[names[type2]]))
        plt.plot(points, jaccard, label='{} & {}'.format(names[type1], names[type2]))
    jaccard = []
    s = range(len(curves))
    for percent in points:
        tmp = []
        for i in range(10):
            a = set(np.random.choice(s, int(percent*len(s)), replace=False))
            b = set(np.random.choice(s, int(percent*len(s)), replace=False))
            tmp.append(len(a & b) / len(a | b))
        jaccard.append(np.mean(tmp))
    plt.plot(points, jaccard, label='random baseline')
    plt.legend(loc=4)
    plt.title(concept)


if 0:
    percent = 0.3
    types = [exponential_average, model_skills, exponential_average_difficulty, exponential_average_time, exponential_average_difficulty_time, model_skills_time]
    types_name = [names[t] for t in types]
    mastereds = {}
    for type in types:
        cache_name = '{}-{}-{}'.format(data, concept, type.__name__)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        df = thresholds(curves, cache=cache_name)
        student_count = len(curves)
        trashold = ((~df.isnull()).sum(axis=0) < student_count * percent).argmax()
        mastered = set(df.index[~df[trashold].isnull()])
        mastereds[names[type]] = mastered
    df = pd.DataFrame(index=types_name, columns=types_name, dtype=float)
    for type in types_name:
        for type2 in types_name:
            df.loc[type, type2] = len(mastereds[type] & mastereds[type2]) / len(mastereds[type] | mastereds[type2])
    print(df)
    plt.title('{} - {}%'.format(concept, percent * 100))
    sns.heatmap(df, vmin=0.)

#### mastery metrics correlations
if 0:
    alpha = 0.9
    for i, concept in enumerate(['numbers', 'addition', 'multiplication', 'subtraction',]):
        types = [exponential_average, model_skills, exponential_average_time, model_skills_time]
        df = pd.DataFrame()
        for type in types:
            cache_name = '{}-{}-{}-{}'.format(data, concept, type.__name__, alpha)
            t = partial(type, exp=alpha)
            curves = get_mastery_curves(data, concept, t, cache=cache_name)
            df[names[type]] = get_order(curves)

        print(df.corr(method='spearman'))
        plt.subplot(2, 2, i + 1)
        sns.heatmap(df.corr(method='spearman'), annot=True, vmax=1, vmin=0.2)
        plt.title(concept)


#### mastery metrics
if 0:
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
    plt.title(concept)


#### alpha
if 0:
    for alpha in [0.7, 0.8, 0.9, 0.95]:
        type = partial(exponential_average, exp=alpha)
        cache_name = '{}-{}-exp{}'.format(data, concept, alpha)
        curves = get_mastery_curves(data, concept, type, cache=cache_name)
        df = thresholds(curves, cache=cache_name)

        plt.plot(df.columns[low_limit:], [df[point].mean() for point in df.columns[low_limit:]], label=alpha)
        plt.title(concept)

# plt.xlabel('threshold')
plt.legend(loc=4)

plt.show()
