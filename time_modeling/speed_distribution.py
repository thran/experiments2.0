from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from models.model import ItemAvgModel, StudentAvgModel, AvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel, \
    TimeEloHierarchicalModel, TimePriorCurrentModel, TimeConcepts
from utils import data as d
from utils.evaluator import Evaluator
import matplotlib.pylab as plt

from utils.model_comparison import compare_models
from utils.runner import Runner
from utils.utils import grid_search, enumerate_df

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data_all = d.Data("../data/matmat/2016-06-27/answers.pd")
data_all.trim_times()
data_long = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data_long.trim_times()
data = data_all

time_model = BasicTimeModel(alpha=0.6, beta=0.1, K=0.25)
model = TimeCombiner(AvgModel(), time_model)


skills = defaultdict(lambda: [])

def update(student, item):
    skills[student].append(time_model.get_skill(student))

model.after_update_callback = update

Runner(data, model).run(force=True)

# skills = time_model.get_skills(data.get_students())
# sns.distplot(skills)

length = 200
avg_skill = np.zeros((len(skills), length))

if True:
    for i, (student, skill) in enumerate(skills.items()):
        if len(skill) > length:
            avg_skill[i] = np.array(skill[:length])
plt.plot(range(length), avg_skill.mean(axis=0))

if False:
    for student, skill in skills.items():
        if len(skill) > length:
            average = moving_average(skill, 10)
            plt.plot(range(len(average)), average, alpha=0.5)
plt.show()
