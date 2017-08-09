import json
import os
from itertools import product
from functools import reduce
from hashlib import sha1

import numpy as np

from utils import  evaluator
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.cm as cm


def grid_search(data, model, model_params, parameters, metric="rmse", plot_axes=None, time=False):
    params = parameter_grid(parameters)
    print("Combinations:", reduce(lambda x,y: x*y, (map(len,parameters.values())), 1))
    df = pd.DataFrame(columns=list(parameters.keys()) + [metric])
    for p in params:
        model_params.update(p)
        if not time:
            df.loc[len(df)] = list(p.values()) + [evaluator.Evaluator(data, model(**model_params)).get_report(only_train=False)[metric]]
        else:
            df.loc[len(df)] = list(p.values()) + [evaluator.Evaluator(data, model(**model_params)).get_report(only_train=False)['time'][metric]]

    if plot_axes:
        if type(plot_axes) is not list:
            print(df)
            plt.plot(parameters[plot_axes], df[metric])
        else:
            results = df.pivot(*plot_axes)[metric]
            results.sort_index(ascending=False, inplace=True)
            sns.heatmap(results, cmap=cm.get_cmap('viridis'))


def parameter_grid(p):
    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params


def enumerate_df(df, column_name='enum'):
    df[column_name] = 1
    df[column_name] = df[column_name].cumsum()
    return df


def sigmoid(x, c = 0):
    return c + (1 - c) / (1 + np.exp(-x))


def make_hash(s, length=20):
    return sha1(str(s).encode()).hexdigest()[:length]

class Cache:
    def __init__(self, type='pandas', cache_name_var='cache', dir='cache', hash=False):
        self.type = type
        self.cache_name_var = cache_name_var
        self.dir = dir
        self.hash = hash
    def __call__(self, f):
        def wrapper(*args, **kwargs):
            if self.cache_name_var not in kwargs:
                print('skip',kwargs)
                return f(*args, **kwargs)

            extension = '?'
            if self.type == 'pandas':
                extension = 'pd'
            if self.type == 'json':
                extension = 'json'

            cache_name = kwargs[self.cache_name_var]
            if self.hash:
                cache_name = make_hash(cache_name)
            filename = os.path.join(self.dir, f.__name__ + '-' + cache_name + '.' + extension)
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            if os.path.exists(filename):
                # print("read", filename)
                if self.type == 'pandas':
                    return pd.read_pickle(filename)
                if self.type == 'json':
                    return json.load(open(filename))
                return

            value = f(*args, **kwargs)
            if self.type == 'pandas':
                value.to_pickle(filename)
            if self.type == 'json':
                json.dump(value, open(filename, 'w'))

            # print("write", filename)
            return value
        return wrapper
