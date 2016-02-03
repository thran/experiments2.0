from collections import defaultdict

from matplotlib.colors import ListedColormap
from pandas.tseries.offsets import Minute
from utils.data import Data
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def filter_answers(data, skill):
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
    answers.loc[answers["answer"].isnull(), "answer"] = "#-1#"
    next = answers.groupby(["item", "student"])["answer"].shift(-1)
    answers.loc[~next.isnull(), "next_same"] = answers["answer"] == next
    answers.loc[answers["answer"] == "#-1#", "answer"] = np.nan
    answers["next_correct"] = answers.groupby(["item", "student"])["correct"].shift(-1)
    answers["next_correct_global"] = answers.groupby(["student"])["correct"].shift(-1)


def last_in_session(answers):
    answers["timestamp"] = pd.to_datetime(answers["timestamp"])
    answers["next_timestamp"] = answers.groupby("student")["timestamp"].shift(-1)
    answers["last_in_session"] = answers["next_timestamp"].isnull() | (answers["next_timestamp"] - answers["timestamp"] > pd.Timedelta(Minute(30)))
    # print(answers.loc[:, ["student", "timestamp", "next_timestamp", "last_in_session"]])

def get_stats(data, context, system="matmat"):
    answers = filter_answers(data, context)
    next_item(answers)
    df = pd.DataFrame(columns= ["system", "context", "classification", "statistics", "value"])
    answers = tag_answers(answers)
    for cl, value in (answers.groupby("class").apply(len) / len(answers)).items():
        df.loc[len(df)] = (system, context, cl, "freq", value)
    for cl, value in (answers.groupby("class").apply(len) / len(answers[answers["class"] != "correct"])).items():
        df.loc[len(df)] = (system, context, cl, "wfreq", 0 if cl == "correct" else value)
    for cl, value in (answers.groupby("class")["response_time"].median()).items():
        df.loc[len(df)] = (system, context, cl, "rtime", value)
    for cl, value in (answers.groupby("class")["next_correct"].mean()).items():
        df.loc[len(df)] = (system, context, cl, "successN", value)
    for cl, value in (answers.groupby("class")["next_correct_global"].mean()).items():
        df.loc[len(df)] = (system, context, cl, "successG", value)
    for cl, value in (answers.groupby("class")["last_in_session"].mean()).items():
        df.loc[len(df)] = (system, context, cl, "leave", value)
    for cl, value in (answers.groupby("class")["next_same"].apply(lambda d: 0 if d.sum() == 0 else d.mean())).items():
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


data = Data("../data/matmat/2016-01-04/answers.pd")
# concepts = ["numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1"]
concepts = ["numbers", "addition", "subtraction", "multiplication", "division"]
results = pd.concat([get_stats(data, concept) for concept in concepts])

print(results)
results.to_csv("results/wrong answers/matmat.csv", sep=";", index=False)
plt.show()