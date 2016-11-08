import numpy as np
import pandas as pd
import seaborn as sns
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import ItemAvgModel, StudentAvgModel, AvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel, \
    TimeEloHierarchicalModel, TimePriorCurrentModel, TimeConcepts
from utils import data as d
from utils.evaluator import Evaluator
import matplotlib.pylab as plt

from utils.model_comparison import compare_models
from utils.utils import grid_search, enumerate_df


def grid_search_K():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimeConcepts(concepts=concepts, **kwargs)),
                {"alpha": 0.4, "beta": 0.05},
                {"K": np.arange(0, 0.5, 0.05)},
                plot_axes='K', time=True,
                )

def grid_search_Ks():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimePriorCurrentModel(**kwargs)),
                {"alpha": 0.6, "beta": 0.1},
                {"KC": np.arange(0.1, 0.7, 0.1),"KI": np.arange(0.1, 0.7, 0.1)},
                plot_axes=['KI', 'KC'], time=True,
                )


def grid_search_AB():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimeConcepts(concepts=concepts, **kwargs)),
                {"K": 0.25}, {
                    "alpha": np.arange(0.2, 0.7, 0.2),
                    "beta": np.arange(0.02, 0.1, 0.02),
                }, plot_axes=['alpha', 'beta'], time=True,
                )


def grid_search_AB2():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimePriorCurrentModel(**kwargs)),
                {"KI": 0.3, 'KC': 0.3}, {
                    "alpha": np.arange(0.2, 1.1, 0.2),
                    "beta": np.arange(0.02, 0.2, 0.02),
                }, plot_axes=['alpha', 'beta'], time=True,
                )


def skill_vs_speed(prediction_mode, time_model, data):
    model = TimeCombiner(prediction_mode, time_model)
    Evaluator(data, model).get_report(force_run=True)
    students = data.get_students()
    skills = prediction_mode.get_skills(students)
    fastness = time_model.get_skills(students)
    sns.jointplot(pd.Series(skills), pd.Series(fastness), kind='kde', space=0).set_axis_labels("skill", "speed")


def learning(data, length=99, measures=None):
    if measures is None:
        measures = ['correct']

    data['response_time_log'] = np.log(data['response_time'])

    grouped = data.groupby(['student'])
    # grouped = data.groupby(['student', 'skill_lvl_1'])
    data = grouped.apply(enumerate_df)
    points = range(1, length + 1)

    plt.subplot(1 + len(measures), 1, 1)
    plt.bar(points[:-1], [(data['enum'] == p).sum() for p in points[:-1]])
    plt.ylabel('User count')
    for i, measure in enumerate(measures):
        plt.subplot(1 + len(measures), 1, i + 2)
        plt.plot(points, [data[data['enum'] == p][measure].mean() for p in points])
        plt.xlabel('# answers')
        plt.ylabel(measure)


data = d.Data("../data/matmat/2016-06-27/answers.pd")
data.trim_times()

data_long = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data_long.trim_times()

skills = pd.read_csv('../data/matmat/2016-06-27/skills.csv', index_col='id')
items = pd.read_csv('../data/matmat/2016-06-27/items.csv')
items = items.join(skills, on='skill_lvl_1')
concepts = {}
for concept in items['name'].unique():
    concepts[concept] = list(items.loc[items['name'] == concept, 'id'])

filters = {
    k: lambda df, v=v: df[df['item'].isin(v)] for k, v in concepts.items()
}

compare_models(data, [
    TimeCombiner(AvgModel(), TimeAvgModel()),
    TimeCombiner(AvgModel(), TimeItemAvgModel()),
    TimeCombiner(AvgModel(), TimeStudentAvgModel()),
    TimeCombiner(AvgModel(), BasicTimeModel(alpha=0.6, beta=0.1, K=0.25)),
    TimeCombiner(AvgModel(), TimePriorCurrentModel(alpha=0.4, beta=0.04, KC=0.3, KI=0.3)),
    TimeCombiner(AvgModel(), TimePriorCurrentModel(alpha=0.4, beta=0.04, KC=0.4, KI=0.4, first_level=False)),
    TimeCombiner(AvgModel(), TimeConcepts(alpha=0.4, beta=0.05, K=0.25, concepts=concepts)),
    TimeCombiner(AvgModel(), TimeEloHierarchicalModel()),
], dont=0, answer_filters=filters)


# learning(data.get_dataframe_all().join(pd.read_csv('../data/matmat/2016-06-27/items.csv'), on='item', rsuffix='_item'),
#          measures=['correct', 'response_time', 'response_time_log'])


# skill_vs_speed(
#     EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
#     BasicTimeModel(alpha=0.6, beta=0.1, K=0.25),
#     data
# )


# grid_search_K()
# grid_search_Ks()
# grid_search_AB()
# grid_search_AB2()


# model = TimeCombiner(ItemAvgModel(), TimeEloHierarchicalModel())
# Evaluator(data, model).get_report(force_run=True)
# Evaluator(data, model).brier_graphs(time=True)
# Evaluator(data, TimeCombiner(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), BasicTimeModel(alpha=0.6, beta=0.1, K=0.25))).brier_graphs(time=True)
# Evaluator(data, model).brier_graphs(time_raw=True)
plt.show()
