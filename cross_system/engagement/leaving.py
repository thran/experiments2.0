import os

import pandas as pd

from utils.data import Data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.runner import Runner


def filter_users(df, min_answer_count=11):
    if not min_answer_count:
        return df
    users = pd.DataFrame(index=df['user'].unique())
    users['answers'] = df.groupby('user').apply(len)
    return df[df['user'].isin(users[users['answers'] >= min_answer_count].index)]

def rolling_success_diff(answers, last_count=4, filters=None, only_last=True):
    if filters is None:
        filters = [None]

    data = []
    for filter in filters:
        df = filter_users(answers, min_answer_count=filter)
        for df in df.groupby('user'):
            df = df[1]
            mean = df['correct'].mean()
            if len(df) < last_count:
                continue
            for x in df['correct'].rolling(last_count, last_count).mean():
                if np.isnan(x):
                    continue
                if not only_last:
                    data.append([np.round(x - mean, 1), filter, 0])
            if not only_last:
                data[-1][-1] = 1
            else:
                data.append([x - mean, filter, 1])
    df = pd.DataFrame(data, columns=['rolling_success_diff', 'min_answers', 'leave'])
    if not only_last:
        sns.pointplot(data=df, x='rolling_success_diff', y='leave', hue='min_answers').set(ylim=(0, 0.2))
    else:
        for filter in filters:
            sns.distplot(df.loc[df['min_answers'] == filter, 'rolling_success_diff'], label=str(filter))
        plt.legend(loc=1)
    return df

# data_set = 'matmat'
# data_set = 'umimecesky-doplnovacka'
# data_set = 'slepemapy-random_parts'
data_set = 'anatom'
last_count = 4

df = pd.read_csv('../../data/engagement/{}.csv'.format(data_set))
# rolling_success_diff(df, last_count, filters=[0, 21])
# plt.title(data_set + " - last " + str(last_count))



plt.show()
