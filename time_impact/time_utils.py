import os

import numpy as np
import pandas as pd

from models.eloPriorCurrent import EloPriorCurrentModel
from utils.runner import Runner
from utils import data as d


def rolling_success(values, initial_value=0, exp=0.9):
    current = initial_value
    results = []
    for value in values:
        current = current * exp + (1 - exp) * value
        results.append(current)
    return results


def sigmoid(x, c = 0):
    return c + (1 - c) / (1 + np.exp(-x))


def get_difficulties(answers, data=None, model=None, force=False, name='difficulty'):
    if data and model:
        runner = Runner(data, model)
        file_name = '../cache/difficulties_{}.pd'.format(runner._hash)
    else:
        data = d.Data("../data/matmat/2016-11-28/answers.pd")
        model = EloPriorCurrentModel(KC=2, KI=0.5)
        runner = Runner(data, model)
        file_name = '../cache/difficulties_matmat.pd'
    if os.path.exists(file_name) and not force:
        difficulties = pd.read_pickle(file_name)
    else:
        items = answers['item'].unique()
        runner.run(force=True)
        difficulties =  pd.Series(data=model.get_difficulties(items), index=items, name=name)
        difficulties.to_pickle(file_name)

    return difficulties