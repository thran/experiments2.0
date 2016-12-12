from collections import defaultdict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.time_models import TimePriorCurrentModel
from time_impact.time_utils import get_difficulties, rolling_success
from utils import data as d
from utils.data import LinearDrop, items_in_concept, \
    TimeLimitResponseModificator


def get_order(answers, metrics, min_answers=50):
    sessions = answers.groupby('session').apply(len)
    sessions = sessions[sessions >= min_answers]

    metric = answers["correct"]

    ends = defaultdict(lambda: [])
    for session in sessions.index:
        answer_index = answers[answers['session'] == session].index
        for name, metric in metrics.items():
            a = metric.loc[answer_index]
            success = rolling_success(a)
            end = success[-1]
            ends[name].append(end)

    return pd.DataFrame.from_dict(ends)


data = d.Data("../data/matmat/2016-11-28/answers.pd")
data.trim_times()
answers_all = data.get_dataframe_all()

difficulties = get_difficulties(answers_all)
time_intensity = get_difficulties(answers_all,
    model=TimePriorCurrentModel(alpha=0.4, beta=0.04, KC=0.3, KI=0.3, first_level=False),
    data=data, name="time_intensity")

# answers_all = answers_all[~answers_all['answer'].isnull()]  # filter missing answers

output = defaultdict(lambda: '')
for concept in ['numbers', 'addition', 'subtraction', 'multiplication', 'division']:
    answers = answers_all[answers_all['item'].isin(items_in_concept(data, concept))]

    time_median = answers['response_time'].median()

    for transformation, answers_transform in {
        "C + lin T": LinearDrop(time_median * 2).modify(answers.copy()),
        "C + binary T": TimeLimitResponseModificator([(time_median, 0.5)]).modify(answers.copy()),
    }.items():
        metrics = {'C': answers['correct'], transformation: answers_transform['correct']}
        orders = get_order(answers, metrics)
        # sns.jointplot('C', transformation, data=orders, kind='kde')

        output[transformation] += '{:<15} -- spearman: {:.3f}, kendall: {:.3f}, sessions: {:>4}, time median: {:.1f}\n'.format(
            concept,
            orders.corr('spearman')['C'][transformation],
            orders.corr('kendall')['C'][transformation],
            len(orders.index),
            time_median,
        )

        # for metric in metrics:
        #     orders[metric] = orders[metric].argsort().argsort()
        # sns.jointplot('C', transformation, data=orders)

for k, v in output.items():
    print(k)
    print(v)

plt.show()
