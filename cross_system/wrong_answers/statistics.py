import os
from collections import defaultdict

from matplotlib.colors import ListedColormap
from pandas.tseries.offsets import Minute
from utils.data import Data, convert_slepemapy, convert_prosoapp
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def filter_answers(data, skill=None, prosoapps=False):
    answers = data.get_dataframe_all()
    if skill is not None and skill is not "all":
        if prosoapps:
            skill = skill.split("-")
            items = data.get_items_df(filename="flashcards.csv", with_skills=False)
            items = items[(items["term_type"] == skill[1]) & (items["context_name"] == skill[0])]
        else:
            pk, level = data.get_skill_id(skill)
            items = data.get_items_df()
            items = items[items["skill_lvl_" + str(level)] == pk]
        answers = pd.DataFrame(answers[answers["item"].isin(items.index)])
    last_in_session(answers)
    if prosoapps:
        answers = answers[answers["guess"] == 0].copy()
    return answers


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
    answers["timestamp"] = pd.to_datetime(answers["time"])
    answers["next_timestamp"] = answers.groupby("student")["timestamp"].shift(-1)
    answers["last_in_session"] = answers["next_timestamp"].isnull() | (answers["next_timestamp"] - answers["timestamp"] > pd.Timedelta(Minute(30)))
    # print(answers.loc[:, ["student", "timestamp", "next_timestamp", "last_in_session"]])

def get_stats(data, context, system="matmat", plot=False):
    print(context)

    def zero_or_mean(df):
        if df.sum() == 0:
            return 0
        else:
            return df.mean()

    answers = filter_answers(data, context, prosoapps=system!="matmat")
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

    if plot:
        pivot = df.pivot("statistics", "classification", "value")
        plt.figure()
        plt.title("{} - {}".format(system, context))
        pivot = pivot.loc[["freq", "wfreq", "rtime", "successN", "successG", "leave", "repetition"], ["correct", "topcwa", "cwa", "missing", "other"]]
        sns.heatmap(pivot, annot=True, vmax=1)
    return df


def radek_plot(data):
    COLORS = sns.color_palette()
    if type(data) is str:
        data = pd.read_csv(data, sep = ";")
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
    data = Data("../../data/matmat/2016-01-04/answers.pd")
    # concepts = ["numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1"]
    concepts = ["all", "numbers", "addition", "subtraction", "multiplication", "division"]
    results = pd.concat([get_stats(data, concept) for concept in concepts])

    print(results)
    results.to_csv("results/matmat.csv", sep=";", index=False)

if False:
    # convert_slepemapy("../data/slepemapy/2016-ab-target-difficulty/answers.csv")
    data = Data("../../data/slepemapy/2016-ab-target-difficulty/answers.pd")
    concepts = ["all", "United States-state", "United States-city", "Czech Rep.-river", "Czech Rep.-mountains", "Europe-state", "Africa-state", "World-state"]
    # concepts = ["all"]
    results = pd.concat([get_stats(data, concept, system="slepemapy") for concept in concepts])

    print(results)
    results.to_csv("results/slepemapy.csv", sep=";", index=False)

if False:
    # convert_prosoapp("../../data/anatom/2016-02-11/answers.csv")
    data = Data("../../data/anatom/2016-02-11/answers.pd")
    concepts = ["all"]
    results = pd.concat([get_stats(data, concept, system="anatom") for concept in concepts])

    print(results)
    results.to_csv("results/anatom.csv", sep=";", index=False)

# radek_plot("results/{}.csv".format("anatom"))
# radek_plot("results/{}.csv".format("slepemapy"))

if True:
    normalize = False

    matmat = pd.read_csv("results/{}.csv".format("matmat"), sep = ";")
    slepemapy = pd.read_csv("results/{}.csv".format("slepemapy"), sep = ";")
    anatom = pd.read_csv("results/{}.csv".format("anatom"), sep = ";")
    results = pd.concat([matmat, slepemapy, anatom])

    STATS = list(results["statistics"].unique())
    STATS.remove('freq') # nuda

    if normalize:
        results = results[results["context"] == "all"]
        for stat in STATS:
            for system in results["system"].unique():
                filter = (results["system"] == system) & (results["statistics"] == stat)
                value = results.loc[filter & (results["classification"] == "correct"), "value"].iloc[0]
                if value:
                    results.loc[filter, "value"] /= value

    if False:
        for i, stat in enumerate(STATS):
            plt.subplot(3, 2, i+1)
            plt.title(stat)
            sns.pointplot(data=results[results["statistics"] == stat], x="classification", y="value", hue="system", order=['correct', 'topcwa', 'cwa', 'other', 'missing'])

    if True:
        results2 = results[results["context"] == "all"]
        plt.subplot(121)
        sns.barplot(data=results2[results2["statistics"] == "wfreq"], x="classification", y="value", hue="system", order=['topcwa', 'cwa', 'other', 'missing'], )
        plt.ylim(0, 0.5)

    if True:
        plt.subplot(122)
        results2 = results[results["system"] == "matmat"]
        print(results["context"].unique())
        print(results)
        sns.barplot(data=results2[results2["statistics"] == "wfreq"], x="classification", y="value", hue="context",
                    order=['topcwa', 'cwa', 'other', 'missing'],
                    hue_order=["numbers", "addition", "subtraction", "multiplication", "division"]
                    )

    if False:
        STATS.remove('wfreq')
        STATS.remove('repetition')

        for i, stat in enumerate(STATS):
            plt.subplot(2, 2, i+1)
            plt.title(stat)
            sns.pointplot(data=results[results["statistics"] == stat], x="classification", y="value", hue="system", order=['correct', 'topcwa', 'cwa', 'other', 'missing'])

    plt.show()

# current_palette = sns.color_palette()
# sns.palplot(current_palette)
# print([[c * 255 for c in p] for p in current_palette])