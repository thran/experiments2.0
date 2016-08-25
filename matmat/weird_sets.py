from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from models.eloHierarchical import EloHierarchicalModel
from models.model import ItemAvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel
from utils import data as d
import matplotlib.pylab as plt

from utils.evaluator import Evaluator
from utils.runner import Runner
from utils.utils import enumerate_df


def prepare_data():
    data = d.Data("../data/matmat/2016-06-27/answers.pd")
    data.trim_times()


    df = data.get_dataframe_all()
    df = df.groupby(['student']).apply(enumerate_df)

    model = ItemAvgModel()
    Runner(data, model).run(force=True)

    items = data.get_items()
    difficulties = pd.Series(index=items, data=model.get_difficulties(items), name='difficulty')
    df = df.join(difficulties, on='item')
    df.to_pickle('../cache/weird_sets.pd')

df = pd.read_pickle('../cache/weird_sets.pd')
df = df.join(pd.read_csv('../data/matmat/2016-06-27/items.csv'), on='item', rsuffix='_item')
skills = pd.read_csv('../data/matmat/2016-06-27/skills.csv', index_col='id')

# print(df['skill_lvl_1'].unique())
for i, skill in enumerate(sorted([0, 26, 367, 2, 209, 151])):
    plt.subplot(3, 2, i+1)

    dff = df[df['skill_lvl_1'] == skill] if skill else df
    points = range(1, 51)
    plt.plot(points, [dff.loc[dff['enum'] == p, 'difficulty'].mean() for p in points])
    plt.title(skills.loc[skill, 'name'] if skill else 'all')
    plt.ylabel('Avg. item difficulty (1 - success rate)')
    plt.xlabel('# answer')


plt.show()
