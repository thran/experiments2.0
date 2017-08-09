from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from utils.data import Data

if 0:
    systems = ['slepemapy', 'matmat', 'anatom']
    bins = 10
    answers = defaultdict(lambda: [])
    time_in_system = defaultdict(lambda: [])

    for system in systems:
        data = Data('data/{}/answers.pd'.format(system), filter=(0, 10)).get_dataframe_all()

        data.loc[data['response_time'] > 30, 'response_time'] = 30
        students = data.groupby('student')['response_time'].median()
        categories = pd.qcut(students, bins, labels=range(bins))


        for c in range(bins):
            selected = data[data['student'].isin(students[categories == c].index)]
            answers[system].append(selected.groupby('student').apply(len).mean())
            time_in_system[system].append(selected.groupby('student')['response_time'].sum().mean())


    plt.subplot(1, 2, 1)
    plt.title('number of answers')
    for system in systems:
        plt.plot(range(bins), answers[system], label=system)
    plt.legend()
    plt.xlabel('learner bin based on median of response time')
    plt.ylabel('answers')
    plt.ylim(ymin=0)

    plt.subplot(1, 2, 2)
    plt.title('time in system')
    for system in systems:
        plt.plot(range(bins), time_in_system[system], label=system)
    plt.legend()
    plt.xlabel('learner bin based on median of response time')
    plt.ylabel('seconds')
    plt.ylim(ymin=0)


if 0:
    system = 'slepemapy'
    bins = 10
    data = Data('data/{}/answers.pd'.format(system), filter=(0, 10)).get_dataframe_all()
    data['correct'] = data['correct'].astype(int)
    categories = pd.qcut(data['response_time'], 10, labels=range(10))

    results = {
        'current': [],
        'next': [],
        'next_correct': [],
        'next_incorrect': [],
    }

    data['correct_next'] = data.groupby(['student', 'item'])['correct'].shift(-1)
    for c in range(bins):
        selected = data[categories == c]
        results['current'].append(selected['correct'].mean())
        results['next'].append(selected['correct_next'].mean())
        results['next_correct'].append(selected[selected['correct'] == 1]['correct_next'].mean())
        results['next_incorrect'].append(selected[selected['correct'] == 0]['correct_next'].mean())

    for key, values in results.items():
        plt.plot(range(bins), values, label=key)
    plt.legend()


if 1:
    system = 'matmat'
    bins = 10
    data = Data('data/{}/answers.pd'.format(system), filter=(0, 10)).get_dataframe_all()
    items = pd.read_csv('data/matmat/items.csv', index_col='id')
    group = 'skill_lvl_1'
    data = data.join(items[group], on='item')
    data['correct'] = data['correct'].astype(int)
    categories = pd.qcut(data['response_time'], 10, labels=range(10))

    results = {
        'current': [],
        'next': [],
        'next_correct': [],
        'next_incorrect': [],
    }

    data['correct_next'] = data.groupby(['student'])['correct'].shift(-1)
    for c in range(bins):
        selected = data[categories == c]
        results['current'].append(selected['correct'].mean())
        results['next'].append(selected['correct_next'].mean())
        results['next_correct'].append(selected[selected['correct'] == 1]['correct_next'].mean())
        results['next_incorrect'].append(selected[selected['correct'] == 0]['correct_next'].mean())

    for key, values in results.items():
        plt.plot(range(bins), values, label=key)
    plt.legend()


plt.show()