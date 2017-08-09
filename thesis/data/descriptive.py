from utils.data import convert_slepemapy, Data
import pandas as pd
import matplotlib.pylab as plt
from utils import data
import seaborn as sns

datasets = {
    'matmat': data.Data('matmat/answers.pd'),
    'geography': data.Data('slepemapy/answers.pd'),
    'czech': data.Data('umimecesky/answers.pd'),
    'anatomy': data.Data('anatom/answers.pd'),

    'mathgarden-addition': data.Data('mathgarden/addition.pd'),
    'mathgarden-multiplication': data.Data('mathgarden/multiplication.pd'),
    'mathgarden-subtraction': data.Data('mathgarden/subtraction.pd'),
}


def drop_off(data, size=200):
    counts = data.groupby("student").size()
    r = range(1, size)
    return r, [sum(counts.values >= count) / len(counts) for count in r]


def drop_offs():
    for name, df in datasets.items():
        x, y = drop_off(df.get_dataframe_all())
        plt.plot(x, y, label=name)
    plt.legend()
    plt.show()


def time_hists(limit=30):
    for name, df in datasets.items():
        times = df.get_dataframe_all()['response_time']
        times = times[times <= limit]
        sns.distplot(times, label=name, hist=False)
    plt.legend()
    plt.show()

time_hists()
