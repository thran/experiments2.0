import os
from itertools import product
from functools import reduce, wraps
from utils import  evaluator
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

def grid_search(data, model, model_params, parameters, metric="rmse", plot_axes=None, time=False):
    params = parameter_grid(parameters)
    print("Combinations:", reduce(lambda x,y: x*y, (map(len,parameters.values())), 1))
    df = pd.DataFrame(columns=list(parameters.keys()) + [metric])
    for p in params:
        model_params.update(p)
        if not time:
            df.loc[len(df)] = list(p.values()) + [evaluator.Evaluator(data, model(**model_params)).get_report()[metric]]
        else:
            df.loc[len(df)] = list(p.values()) + [evaluator.Evaluator(data, model(**model_params)).get_report()['time'][metric]]

    if plot_axes:
        if type(plot_axes) is not list:
            print(df)
            plt.plot(parameters[plot_axes], df[metric])
        else:
            results = df.pivot(*plot_axes)[metric]
            results.sort_index(ascending=False, inplace=True)
            sns.heatmap(results)


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


def cache_pandas(f, cache_name_var='cache', dir='cache'):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if cache_name_var not in kwargs:
            return f(*args, **kwargs)

        filename = os.path.join(dir, f.__name__ + '-' + kwargs[cache_name_var] + '.pd')
        if not os.path.exists(dir):
            os.makedirs(dir)
        if os.path.exists(filename):
            # print("read", filename)
            return pd.read_pickle(filename)

        value = f(*args, **kwargs)
        value.to_pickle(filename)
        # print("write", filename)
        return value

    return wrapper