from collections import defaultdict

from matplotlib.colors import ListedColormap
from pandas.tseries.offsets import Minute
from utils.data import Data, convert_slepemapy
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def filter_answers(data, skill, slepemapy=False):
    if slepemapy:
        skill = skill.split("-")
        items = data.get_items_df(filename="flashcards.csv", with_skills=False)
        items = items[(items["term_type"] == skill[1]) & (items["context_name"] == skill[0])]
    else:
        pk, level = data.get_skill_id(skill)
        items = data.get_items_df()
        items = items[items["skill_lvl_" + str(level)] == pk]
    answers = data.get_dataframe_all()
    last_in_session(answers)
    return pd.DataFrame(answers[answers["item"].isin(items.index)])


def tag_answers(answers):
    answers.loc[:, "class"] = "correct"
    wrong = pd.DataFrame(answers[answers["correct"] != 1])
    wrong.loc[:, "class"] = "other"

    counts = wrong[~wrong["answer"].isnull()].groupby("item").apply(len)
    for (item, answer), count in wrong.groupby(["item", "answer"]).apply(len).items():
        if count > 1 and 0.05 * counts[item] < count:
            wrong.loc[(wrong["item"] == item) & (wrong["answer"] == answer), "class"] = "cwa"

    bests = defaultdict(lambda: (0, 0))
    for (item, answer), count in wrong.groupby(["item", "answer"]).apply(len).items():
        if bests[item][0] < count:
            bests[item] = count, answer
    for item, (count, answer) in bests.items():
        wrong.loc[(wrong["item"] == item) & (wrong["answer"] == answer), "class"] = "topcwa"

    wrong.loc[wrong["answer"].isnull(), "class"] = "missing"
    answers.loc[wrong.index, "class"] = wrong["class"]
    return answers


def next_item(answers):
    answers.loc[answers["answer"].isnull(), "answer"] = -1
    next = answers.groupby(["item", "student"])["answer"].shift(-1)
    answers.loc[~next.isnull(), "next_same"] = answers["answer"] == next
    answers.loc[answers["answer"] == -1, "answer"] = np.nan
    answers["next_correct"] = answers.groupby(["item", "student"])["correct"].shift(-1)
    answers["next_correct_global"] = answers.groupby(["student"])["correct"].shift(-1)


def last_in_session(answers):
    if "last_in_session" in answers.columns:
        return
    answers["timestamp"] = pd.to_datetime(answers["timestamp"])
    answers["next_timestamp"] = answers.groupby("student")["timestamp"].shift(-1)
    answers["last_in_session"] = answers["next_timestamp"].isnull() | (answers["next_timestamp"] - answers["timestamp"] > pd.Timedelta(Minute(30)))
    # print(answers.loc[:, ["student", "timestamp", "next_timestamp", "last_in_session"]])

def get_stats(data, context, system="matmat"):
    print(context)

    def zero_or_mean(df):
        if df.sum() == 0:
            return 0
        else:
            return df.mean()

    answers = filter_answers(data, context, slepemapy=system=="slepemapy")
    next_item(answers)
    df = pd.DataFrame(columns= ["system", "context", "classification", "statistics", "value"])
    answers = tag_answers(answers)
    for cl, value in (answers.groupby("class").apply(len) / len(answers)).items():
        df.loc[len(df)] = (system, context, cl, "freq", value)
    for cl, value in (answers.groupby("class").apply(len) / len(answers[answers["class"] != "correct"])).items():
        df.loc[len(df)] = (system, context, cl, "wfreq", 0 if cl == "correct" else value)
    for cl, value in (answers.groupby("class")["response_time"].median()).items():
        df.loc[len(df)] = (system, context, cl, "rtime", value)
    for cl, value in (answers.groupby("class")["next_correct"].apply(zero_or_mean)).items():
        df.loc[len(df)] = (system, context, cl, "successN", value)
    for cl, value in (answers.groupby("class")["next_correct_global"].apply(zero_or_mean)).items():
        df.loc[len(df)] = (system, context, cl, "successG", value)
    for cl, value in (answers.groupby("class")["last_in_session"].mean()).items():
        df.loc[len(df)] = (system, context, cl, "leave", value)
    for cl, value in (answers.groupby("class")["next_same"].apply(zero_or_mean)).items():
        df.loc[len(df)] = (system, context, cl, "repetition", value)


    pivot = df.pivot("statistics", "classification", "value")
    # print(df)
    # print(pivot)
    # plt.switch_backend('agg')
    plt.figure()
    plt.title("{} - {}".format(system, context))
    pivot = pivot.loc[["freq", "wfreq", "rtime", "successN", "successG", "leave", "repetition"], ["correct", "topcwa", "cwa", "missing", "other"]]
    sns.heatmap(pivot, annot=True, vmax=1)
    plt.savefig("results/wrong answers/{}.png".format(context))
    return df


def radek_plot(filename):
    COLORS = sns.color_palette()
    data = pd.read_csv(filename, sep = ";")
    CL = ['correct', 'topcwa', 'cwa' , 'other', 'missing' ]

    contexts = data.context.unique()
    contexts.sort()

    def select(data, c, cl, stat):
        d = data[(data.context == c) & (data.classification == cl) & (data.statistics == stat)]
        return d.value.iloc[0]

    def do_plot(stat):
        ind = np.arange(len(CL))
        w = 0.15
        for i, c in enumerate(contexts):
            vals = [ select(data, c, cl, stat) for cl in CL ]
            plt.bar(ind + w*i, vals, w, color = COLORS[i])

        plt.xticks(ind + 0.3, CL, rotation=0)
        plt.ylabel(stat)

    STATS = list(data.statistics.unique())
    STATS.remove('freq') # nuda
    for i, stat in enumerate(STATS):
        plt.subplot(3, 2, i+1)
        do_plot(stat)
    plt.legend(contexts, loc = "best")
    plt.show()

if False:
    data = Data("../data/matmat/2016-01-04/answers.pd")
    # concepts = ["numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1"]
    concepts = ["numbers", "addition", "subtraction", "multiplication", "division"]
    results = pd.concat([get_stats(data, concept) for concept in concepts])

    print(results)
    results.to_csv("results/wrong answers/matmat.csv", sep=";", index=False)
    plt.show()

if False:
    # convert_slepemapy("../data/slepemapy/2016-ab-target-difficulty/answers.csv")
    data = Data("../data/slepemapy/2016-ab-target-difficulty/answers.pd")
    concepts = ["Czech Rep.-river", "Czech Rep.-mountains", "Europe-state", "Africa-state", "World-state"]
    results = pd.concat([get_stats(data, concept, system="slepemapy") for concept in concepts])

    print(results)
    results.to_csv("results/wrong answers/slepemapy.csv", sep=";", index=False)
    plt.show()

# radek_plot("results/wrong answers/{}.csv".format("matmat"))
radek_plot("results/wrong answers/{}.csv".format("slepemapy"))