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
d3 = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.LinearDrop(14))

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

def mean_by_skill(data, values, filter_skills=None):
    items = data.get_items_df()
    if filter_skills is not None:
        items = items[items["skill_lvl_1"].isin(filter_skills) | items["skill_lvl_2"].isin(filter_skills)]
    items = items.join(pd.Series(values, name="value"))
    return items.groupby("skill")["value"].mean()


def compare_models(d1, d2, m1, m2, filter_skills=None):
    n1 = isinstance(m1, EloHierarchicalModel)
    n2 = isinstance(m2, EloHierarchicalModel)
    skills = d1.get_skills_df()
    v1 = mean_by_skill(d1, get_difficulty(d1, m1, n1), filter_skills)
    v2 = mean_by_skill(d2, get_difficulty(d2, m2, n2), filter_skills)
    plt.title(v1.corr(v2))
    for k, v in v1.items():
        if not np.isnan(v2[k]) and not np.isnan(v):
            plt.plot(v, v2[k], "bo")
            plt.text(v, v2[k], skills.loc[k, "name"])
    # plt.xlabel("{} - {}".format(m1, d1))
    # plt.ylabel("{} - {}".format(m2, d2))
    plt.xlabel("Difficulty according to the Basic model")
    plt.ylabel("Difficulty according to the Basic model + linerTime")


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

if 1:
    compare_models(d, d3,
        EloPriorCurrentModel(KC=2, KI=0.5),
        EloPriorCurrentModel(KC=2, KI=0.5),
        # EloPriorCurrentModel(KC=2, KI=0.5),
        # EloHierarchicalModel(alpha=0.25, beta=0.02, KC=3.5, KI=2.50),
        # EloHierarchicalModel(alpha=0.80, beta=0.02, KC=1.0, KI=0.75),
        # EloHierarchicalModel(alpha=0.25, beta=0.02),
        # EloConcepts(concepts=concepts),
        # ItemAvgModel(),
    filter_skills=[27]) # all skills [2, 26, 151, 209, 367]
    # n1=True, n2=True)

if 0:
    df = d.get_dataframe_all()
    df = df.join(d.get_items_df(), on="item", lsuffix="item_")
    plot_skill_values(d, df.groupby("skill").size())
    plt.ylim(0, 2000)
    # plot_item_values(d, d.get_dataframe_all().groupby("item").size())


def difficulty_vs_time(data, the_skill, concepts=False):
    data.filter_data(0, 100)
    pk, level = data.get_skill_id(the_skill)
    data.trim_times()
    data.add_log_response_times()
    m = EloPriorCurrentModel(KC=2, KI=0.5)
    items = data.get_items_df()
    items = items[items["visualization"] != "pairing"]
    items = items.join(get_difficulty(data, m))
    items = items.join(pd.Series(data.get_dataframe_all().groupby(["item"])["log_response_time"].mean(), name="log_response_time_mean"))
    items = items[items["skill_lvl_"+str(level)] == pk]

    if concepts:
        skills = data.get_skills_df()
        skills = skills.join(items.groupby("skill_lvl_3")["difficulty"].mean())
        skills = skills.join(items.groupby("skill_lvl_3")["log_response_time_mean"].mean())
        skills = skills[skills.index.isin(items["skill_lvl_3"].unique())]
        for id, skill in skills.iterrows():
            plt.plot(skill["difficulty"], skill["log_response_time_mean"], "ok")
            plt.text(skill["difficulty"], skill["log_response_time_mean"], skill["name"])
    else:
        colors = "rgbyk"
        visualizations = list(items["visualization"].unique())
        for id, item in items.iterrows():
            plt.plot(item["difficulty"], item["log_response_time_mean"], "o", color=colors[visualizations.index(item["visualization"])])
            plt.text(item["difficulty"], item["log_response_time_mean"], item["name"])
        for i, vis in enumerate(visualizations):
            plt.plot(-1, 2, "o", color=colors[i], label=vis)
    plt.xlabel("difficulty according to " + str(m))
    plt.ylabel("mean of log time")
    plt.legend(loc=0)
    plt.title(the_skill)



plt.show()