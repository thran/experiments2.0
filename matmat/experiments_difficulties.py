import random

from models.eloConcepts import EloConcepts
from utils import data, evaluator, utils, runner
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import AvgModel, ItemAvgModel
from models.eloHierarchical import EloHierarchicalModel
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

d = data.Data("../data/matmat/2015-11-20/answers.pd")
concepts = d.get_concepts()
# d_filtered = data.Data("../data/matmat/2015-11-20/answers.pd", filter=(10, 10))
d2 = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.5)]))

def plot_skill_values(data, values):
    colors = ["p", "b", "y", "k", "g", "r"]
    skills = data.get_skills_df()
    items = data.get_items_df()
    skills = skills.join(items.groupby("skill")["skill_lvl_1"].first())
    colors = dict(zip(skills["skill_lvl_1"].unique(), colors))

    skills = skills.join(pd.Series(values, name="value"))
    skills = skills[~skills["value"].isnull()]
    for _, skill in skills.iterrows():
        x = random.random()
        # print(skill["value"], skill["name"])
        plt.plot(x, skill["value"], "o", color=colors[skill["skill_lvl_1"]] if not np.isnan(skill["skill_lvl_1"]) else "k")
        plt.text(x, skill["value"], skill["name"])

def plot_item_values(data, values):
    plot_skill_values(data, mean_by_skill(data, values))

def mean_by_skill(data, values):
    items = data.get_items_df()
    items = items.join(pd.Series(values, name="value"))
    return items.groupby("skill")["value"].mean()


def compare_models(d1, d2, m1, m2, n1=False, n2=False):
    skills = d1.get_skills_df()
    v1 = mean_by_skill(d1, get_difficulty(d1, m1, n1))
    v2 = mean_by_skill(d2, get_difficulty(d2, m2, n2))
    for k, v in v1.items():
        if not np.isnan(v2[k]) and not np.isnan(v):
            plt.plot(v, v2[k], ".")
            plt.text(v, v2[k], skills.loc[k, "name"])
    plt.xlabel("{} - {}".format(m1, d1))
    plt.ylabel("{} - {}".format(m2, d2))


def get_mean_skill(data, model):
    if not hasattr(model, "skill"):
        runner.Runner(data, model).run(force=True)
    return {skill: sum(values.values()) / len(values) for skill, values in model.skill.items()}


def get_difficulty(data, model, normalize=False):
    runner.Runner(data, model).run(force=True)
    if not normalize:
        return pd.Series(model.difficulty, name="difficulty")

    items = data.get_items_df()
    items = items.join(pd.Series(model.difficulty, name="difficulty"))
    skills = get_mean_skill(data, model)
    for skill, value in skills.items():
        if skill == 1:
            items.loc[:, "difficulty"] -= value
        else:
            items.loc[(items["skill_lvl_1"] == skill) | (items["skill_lvl_2"] == skill) | (items["skill_lvl_3"] == skill), "difficulty"] -= value
    return items["difficulty"]

if 0:
    # m = EloPriorCurrentModel(KC=2, KI=0.5)
    m = EloHierarchicalModel(alpha=0.25, beta=0.02)
    # m = ItemAvgModel()
    # m = EloConcepts(concepts=concepts)
    plt.title(m)
    plot_item_values(d, get_difficulty(d, m, normalize=True))
    # plot_skill_values(d, get_mean_skill(d, m))

if 0:
    compare_models(d, d,
        EloPriorCurrentModel(KC=2, KI=0.5),
        # EloPriorCurrentModel(KC=2, KI=0.5),
        # EloHierarchicalModel(alpha=0.25, beta=0.02),
        # EloHierarchicalModel(alpha=0.25, beta=0.02),
        EloConcepts(concepts=concepts),
        # ItemAvgModel(),
    n1=False, n2=False)
    # n1=True, n2=True)

if 1:
    df = d.get_dataframe_all()
    df = df.join(d.get_items_df(), on="item", lsuffix="item_")
    plot_skill_values(d, df.groupby("skill").size())
    plt.ylim(0, 2000)
    # plot_item_values(d, d.get_dataframe_all().groupby("item").size())

plt.show()