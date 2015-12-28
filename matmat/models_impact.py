from algorithms.spectralclustering import SpectralClusterer
from matmat.experiments_difficulties import difficulty_vs_time, get_difficulty
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from utils.data import Data, TimeLimitResponseModificator
from utils.data import compute_corr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import scipy.cluster.hierarchy as hr
import scipy.spatial.distance as dst

from utils.evaluator import Evaluator
from utils.runner import Runner

cache = {}

def compare_models(data1, data2, model1, model2, plot=True):
    if str(data1) + str(model1) not in cache:
        runner1 = Runner(data1, model1)
        runner1.run(force=True)
        cache[str(data1) + str(model1)] = runner1, model1
    else:
        runner1, model1 = cache[str(data1) + str(model1)]

    if str(data2) + str(model2) not in cache:
        runner2 = Runner(data2, model2)
        runner2.run(force=True)
        cache[str(data2) + str(model2)] = runner2, model2
    else:
        runner2, model2 = cache[str(data2) + str(model2)]

    # difficulties
    items = list(set(data1.get_items()) & set(data2.get_items()))
    difficulties = pd.DataFrame(columns=["model1", "model2"], index=items)
    difficulties["model1"] = model1.get_difficulties(items)
    difficulties["model2"] = model2.get_difficulties(items)
    difficulties_corr = difficulties.corr().loc["model1", "model2"]
    if plot:
        plt.subplot(221)
        plt.plot(difficulties["model1"], difficulties["model2"], "k.")
        plt.title("Difficulties: {}".format(difficulties_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    # skills
    students = list(set(data1.get_students()) & set(data2.get_students()))
    skills = pd.DataFrame(index=students, columns=["model1", "model2"])
    skills["model1"] = model1.get_skills(students)
    skills["model2"] = model2.get_skills(students)
    skills_corr = skills.corr().loc["model1", "model2"]
    if plot:
        plt.subplot(222)
        plt.plot(skills["model1"], skills["model2"], "k.")
        plt.title("Skills: {}".format(skills_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    # predictions
    predictions = pd.DataFrame(index=students, columns=["model1", "model2"])
    predictions["model1"] = pd.Series(runner1._log)
    predictions["model2"] = pd.Series(runner2._log)
    predictions_corr = predictions.corr().loc["model1", "model2"]
    if plot:
        plt.subplot(223)
        plt.plot(predictions["model1"], predictions["model2"], "k.")
        plt.title("Predictions: {}".format(predictions_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    return difficulties_corr, skills_corr, predictions_corr


def single_step_time_split(model, value=0.5):
    limits = range(1, 15)
    datas = lambda limit: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(limit, value)]))

    results_d = pd.DataFrame(index=limits, columns=limits, dtype=float)
    results_s = pd.DataFrame(index=limits, columns=limits, dtype=float)
    results_p = pd.DataFrame(index=limits, columns=limits, dtype=float)
    for limit1 in limits:
        for limit2 in limits:
            d, s, p = compare_models(datas(limit1), datas(limit2), model(), model(), plot=False)
            results_d[limit1][limit2] = d
            results_s[limit1][limit2] = s
            results_p[limit1][limit2] = p

    df = pd.DataFrame(columns=["limit", "rmse"])
    for limit in limits:
        r = Evaluator(datas(limit), model()).get_report()
        df.loc[len(df)] = (limit, r["rmse"])

    plt.title(str(model()))
    plt.subplot(221)
    plt.title("Correlations of difficulties")
    sns.heatmap(results_d)
    plt.subplot(222)
    plt.title("Correlations of skills")
    sns.heatmap(results_s)
    plt.subplot(223)
    plt.title("Correlations of predictions")
    sns.heatmap(results_p)
    plt.subplot(224)
    sns.barplot(x="limit", y="rmse", data=df,)


model_flat = lambda: EloPriorCurrentModel(KC=2, KI=0.5)
model_hierarchical = lambda: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
data = Data("../data/matmat/2015-12-16/answers.pd")
data_time_2 = Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(5, 0.5)]))
data_time_2b = Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(7, 0.5)]))

single_step_time_split(model_flat)
plt.show()

if False:
    plt.figure(figsize=(10, 10), dpi=100)
    # compare_models(data, data, model_flat(), model_hierarchical())
    print(compare_models(data_time_2, data_time_2b, model_flat(), model_flat()))
    plt.show()