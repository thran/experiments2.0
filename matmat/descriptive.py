import os

import pandas as pd

from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from utils.data import Data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.runner import Runner


def response_times(data, time_dist=True, mean_times_dist=True):
    data.filter_data()
    data.trim_times()
    data.add_log_response_times()
    df = data.get_dataframe_all()

    if time_dist:
        plt.figure()
        # sns.distplot(df["response_time"], hist=False, bins=30, label="all")
        # sns.distplot(df[df["answer"].isnull()]["response_time"], hist=False, bins=30, label="without answer (null)")
        sns.distplot(df[df["correct"] == True]["response_time"], hist=False, bins=30, label="correct answer")
        sns.distplot(df[df["correct"] == False][~df["answer"].isnull()]["response_time"], hist=False, bins=30, label="wrong answer (not null)")
        plt.title("Response time distribution per answer type")

    if mean_times_dist:
        plt.figure()
        sns.distplot(np.exp(df.groupby("item")["log_response_time"].mean()), hist=False, label="items ({})".format(len(data.get_items())))
        sns.distplot(np.exp(df.groupby("student")["log_response_time"].mean()), hist=False, label="students ({})".format(len(data.get_students())))
        # sns.distplot(np.exp(df[~df["answer"].isnull()].groupby("item")["log_response_time"].mean()), hist=False, label="items - with answer({})".format(len(data.get_items())))
        # sns.distplot(np.exp(df[~df["answer"].isnull()].groupby("student")["log_response_time"].mean()), hist=False, label="students - with answer ({})".format(len(data.get_students())))


        plt.title("Distribution of median time (exp of median of log times)")


def answer_count(data, per_student=True, per_item=True, student_drop_off=True):
    # data.filter_data()
    df = data.get_dataframe_all()

    if per_student:
        plt.figure()
        sns.distplot(df.groupby("student").size(), kde=False, bins=30, label="", hist_kws={"range": [0, 300]})
        plt.xlabel("answer count")
        plt.ylabel("student count")
        plt.title("Answer count distribution per student")

    if per_item:
        plt.figure()
        sns.distplot(df.groupby("item").size(), kde=False, bins=300, label="", hist_kws={"range": [0, 300]})
        plt.xlabel("answer count")
        plt.ylabel("item count")
        plt.title("Answer count distribution per item")

    if student_drop_off:
        plt.figure()
        counts = df.groupby("student").size()
        r = range(1, 100)
        plt.plot(r, [sum(counts.values >= count) / len(counts) for count in r])
        plt.xlabel("answer count")
        plt.ylabel("percentage of students")
        plt.title("Student drop-off")


def success_rate(data, per_student=True, per_item=True):
    data.filter_data()
    df = data.get_dataframe_all()

    if per_student:
        plt.figure()
        sns.distplot(df.groupby("student")["correct"].mean(), kde=False, bins=50, label="", hist_kws={"range": [0, 1]})
        plt.xlabel("success rate")
        plt.ylabel("student count")
        plt.title("Success rate distribution per student")

    if per_item:
        plt.figure()
        sns.distplot(df.groupby("item")["correct"].mean(), kde=False, bins=50, label="", hist_kws={"range": [0, 1]})
        plt.xlabel("success rate")
        plt.ylabel("item count")
        plt.title("Success rate distribution per item")


def pair_grid(data):
    df = data.get_dataframe_all()
    data.trim_times()
    data.add_log_response_times()
    students = pd.DataFrame(index=df["student"].unique())

    students["Success rate"] = df.groupby("student")["correct"].mean()
    # students["avg_response_time"] = df.groupby("student")["response_time"].mean()
    students["Avg. log of response time"] = df.groupby("student")["log_response_time"].mean()
    if False and os.path.exists(os.path.dirname(data._filename) + "/skill_global.pd"):
        students["Skill"] = pd.read_pickle(os.path.dirname(data._filename) + "/skill_global.pd")
    students["Prior skill"] = pd.read_pickle(os.path.dirname(data._filename) + "/skill_prior.pd")
    students["Log of answer count"] = np.log10(df.groupby("student").size())

    # students = students.sample(1000)
    if "slepemapy" in data._filename and False:
        groups = df.groupby("student")["experiment_setup_id"].first()
        students["AB group"] = ""
        for i, name in ((14, "50%"), (15, "35%"), (16, "20%"), (17, "5%"), ):
            students["AB group"][groups == i] = name
        g = sns.PairGrid(students, hue="AB group", hue_order=["5%", "20%", "35%", "50%", ])
        g = g.map_diag(sns.distplot)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot)
        g = g.add_legend()
    else:
        g = sns.PairGrid(students)
        g = g.map_diag(plt.hist)
        g = g.map_upper(plt.scatter, marker=".")
        g = g.map_lower(sns.kdeplot, shade=False)


data = Data("../data/matmat/2017-03-29/answers.pd")
# data = Data("../data/slepemapy/2016-ab-target-difficulty/answers.pd")
# data.filter_data(10, 10)
items = data.get_items_df()
answers = data.get_dataframe_all()
skills = data.get_skills_df()

response_times(data, time_dist=True, mean_times_dist=True)
# answer_count(data, per_student=True, per_item=True, student_drop_off=True)
# success_rate(data, per_student=True)
# print(data.get_items_df().count())

# pair_grid(data)
plt.show()

if False:
    print('item count', len(items))
    print('answer count', len(answers))
    df = answers.merge(items, left_on='item', right_index=True)
    df = df.merge(skills, left_on='skill_lvl_1', right_index=True)
    print(df.groupby('name_y').size())


if False:
    model = EloPriorCurrentModel(KC=2, KI=0.5)
    # model = EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
    Runner(data, model).run(force=True)
    pd.Series(model.global_skill).to_pickle(os.path.dirname(data._filename) + "/skill_prior.pd")
    # pd.Series(model.skill[1]).to_pickle(os.path.dirname(data._filename) + "/skill_global.pd")



