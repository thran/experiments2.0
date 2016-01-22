from algorithms.spectralclustering import SpectralClusterer
from matmat.experiments_difficulties import difficulty_vs_time, get_difficulty
from models.eloConcepts import EloConcepts
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from models.skipHandler import SkipHandler
from utils.data import Data, TimeLimitResponseModificator, ExpDrop, LinearDrop
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

    difficulties_corr, skills_corr, predictions_corr = 0, 0, 0

    # difficulties
    items = list(set(data1.get_items()) & set(data2.get_items()))
    difficulties = pd.DataFrame(columns=["model1", "model2"], index=items)
    difficulties["model1"] = model1.get_difficulties(items)
    difficulties["model2"] = model2.get_difficulties(items)
    difficulties_corr = difficulties.corr(method="spearman").loc["model1", "model2"]
    if plot:
        plt.subplot(221)
        plt.plot(difficulties["model1"], difficulties["model2"], "k.")
        plt.title("Difficulties: {}".format(difficulties_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    try:
        # skills
        students = list(set(data1.get_students()) & set(data2.get_students()))
        skills = pd.DataFrame(index=students, columns=["model1", "model2"])
        skills["model1"] = model1.get_skills(students)
        skills["model2"] = model2.get_skills(students)
        skills_corr = skills.corr(method="spearman").loc["model1", "model2"]
        if plot:
            plt.subplot(222)
            plt.plot(skills["model1"], skills["model2"], "k.")
            plt.title("Skills: {}".format(skills_corr))
            plt.xlabel(str(model1))
            plt.ylabel(str(model2))
    except AttributeError:
        pass

    # predictions
    predictions = pd.DataFrame(index=students, columns=["model1", "model2"])
    predictions["model1"] = pd.Series(runner1._log)
    predictions["model2"] = pd.Series(runner2._log)
    predictions_corr = predictions.corr(method="spearman").loc["model1", "model2"]
    if plot:
        plt.subplot(223)
        plt.plot(predictions["model1"], predictions["model2"], "k.")
        plt.title("Predictions: {}".format(predictions_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    return difficulties_corr, skills_corr, predictions_corr


def compare_more_models(experiments):
    labels = sorted(experiments.keys())

    results_d = pd.DataFrame(index=labels, columns=labels, dtype=float)
    results_s = pd.DataFrame(index=labels, columns=labels, dtype=float)
    results_p = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for label1 in labels:
        for label2 in labels:
            d, s, p = compare_models(experiments[label1][0](label1), experiments[label2][0](label2),
                                     experiments[label1][1](label1), experiments[label2][1](label2), plot=False)
            results_d[label1][label2] = d
            results_s[label1][label2] = s
            results_p[label1][label2] = p

    df = pd.DataFrame(columns=["labels", "rmse"])
    for label in labels:
        r = Evaluator(experiments[label][0](label), experiments[label][1](label)).get_report()
        df.loc[len(df)] = (label, r["rmse"])

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
    sns.barplot(x="labels", y="rmse", data=df,)


model_flat = lambda label: EloPriorCurrentModel(KC=2, KI=0.5)
model_hierarchical = lambda label: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
data = Data("../data/matmat/2015-12-16/answers.pd")
data_time_2 = Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(5, 0.5)]))
data_time_2b = Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(7, 0.5)]))

value = 0.5
single_step_time_split = {
    limit: (
        lambda limit: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(limit, value)])),
        model_flat
    )
    for limit in range(1, 15)
}
# compare_more_models(single_step_time_split)

limit = 5
single_step_time_split_value = {
    value / 10.: (
        lambda value: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(limit, value)])),
        model_flat
    )
    for value in range(1, 11)
}
# compare_more_models(single_step_time_split_value)

different_models = {
    "flat": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "flat-d": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloPriorCurrentModel()),
    "hierarchical": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "hierarchical-d": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloHierarchicalModel())
}
# compare_more_models(different_models)

different_time_mods = {
    "-": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), model_flat),
    "6-0.5": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(6, 0.5)])), model_flat),
    "6-0.5, 12-0.2": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(6, 0.5), (12, 0.2)])), model_flat),
    "3-0.75, 6-0.5": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(3, 0.75), (6, 0.5)])), model_flat),
    "exp 6-0.9": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=ExpDrop(6, 0.9)), model_flat),
    "exp 6-0.8": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=ExpDrop(6, 0.8)), model_flat),
    "exp 3-0.9": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=ExpDrop(3, 0.9)), model_flat),
    "exp 3-0.8": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=ExpDrop(3, 0.8)), model_flat),
    "lin 10": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=LinearDrop(10)), model_flat),
    "lin 15": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=LinearDrop(15)), model_flat),
    "lin 20": (lambda l: Data("../data/matmat/2015-12-16/answers.pd", response_modification=LinearDrop(15)), model_flat),
}
# compare_more_models(different_time_mods)

with_nans = {
    "flat": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "flat + nan": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: SkipHandler(EloPriorCurrentModel(KC=2, KI=0.5))),
    "hierarchical": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "hierarchical+ nan": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: SkipHandler(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02))),
    "concepts": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: EloConcepts(concepts=data.get_concepts())),
    "concepts + nan": (lambda l: Data("../data/matmat/2015-12-16/answers.pd"), lambda l: SkipHandler(EloConcepts(concepts=data.get_concepts()))),
}

# compare_more_models(with_nans)

plt.show()

if False:
    plt.figure(figsize=(10, 10), dpi=100)
    # compare_models(data, data, model_flat(None), model_hierarchical(None))
    # compare_models(data, data, EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), SkipHandler(EloConcepts(concepts=data.get_concepts())))
    # compare_models(data, data, EloConcepts(concepts=data.get_concepts()), SkipHandler(EloConcepts(concepts=data.get_concepts())))
    # compare_models(data, data, EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), SkipHandler(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)))
    # print(compare_models(data_time_2, data_time_2b, model_flat(None), model_flat(None)))
    plt.show()