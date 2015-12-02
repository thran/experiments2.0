from itertools import product
from functools import reduce
from utils import  evaluator
import pandas as pd
import seaborn as sns

def grid_search(data, model, model_params, parameters, metric="rmse", plot_axes=None):
    params = parameter_grid(parameters)
    print("Combinations:", reduce(lambda x,y: x*y, (map(len,parameters.values())), 1))
    df = pd.DataFrame(columns=list(parameters.keys()) + [metric])
    for p in params:
        model_params.update(p)
        df.loc[len(df)] = list(p.values()) + [evaluator.Evaluator(data, model(**model_params)).get_report()[metric]]


    if plot_axes:
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