import numpy as np
import pandas as pd
import seaborn as sns
from models.eloHierarchical import EloHierarchicalModel
from models.model import ItemAvgModel, StudentAvgModel, AvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel
from utils import data as d
from utils.evaluator import Evaluator
import matplotlib.pylab as plt

from utils.model_comparison import compare_models
from utils.utils import grid_search

data = d.Data("../data/matmat/2016-06-27/answers.pd")
data.trim_times()

data_long = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data_long.trim_times()


def grid_search_K():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), BasicTimeModel(**kwargs)),
                {"alpha": 0.6, "beta": 0.1},
                {"K": np.arange(0, 1, 0.05)},
                plot_axes='K', time=True,
                )

def grid_search_AB():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), BasicTimeModel(**kwargs)),
                {"K": 0.25}, {
                    "alpha": np.arange(0.4, 1.3, 0.2),
                    "beta": np.arange(0.06, 0.2, 0.02),
                }, plot_axes=['alpha', 'beta'], time=True,
                )

def skill_vs_speed(prediction_mode, time_model, data):
    model = TimeCombiner(prediction_mode, time_model)
    Evaluator(data, model).get_report(force_run=True)
    students = data.get_students()
    skills = prediction_mode.get_skills(students)
    fastness = time_model.get_skills(students)
    sns.jointplot(pd.Series(skills), pd.Series(fastness), kind='kde', space=0).set_axis_labels("skill", "speed")


compare_models(data, [
    TimeCombiner(AvgModel(), TimeAvgModel()),
    TimeCombiner(ItemAvgModel(), TimeItemAvgModel()),
    TimeCombiner(StudentAvgModel(), TimeStudentAvgModel()),
    TimeCombiner(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), BasicTimeModel(alpha=0.6, beta=0.1, K=0.25)),
], dont=1)

skill_vs_speed(
    EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
    BasicTimeModel(alpha=0.6, beta=0.1, K=0.25),
    data
)


# grid_search_K()
# grid_search_AB()


# model = TimeCombiner(ItemAvgModel(), TimeStudentAvgModel())
# Evaluator(data, model).get_report(force_run=True)
# Evaluator(data, model).brier_graphs()
# Evaluator(data, model).brier_graphs(time=True)
# Evaluator(data, model).brier_graphs(time_raw=True)
plt.show()
