from itertools import product
from functools import reduce
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