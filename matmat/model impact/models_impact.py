import copy
from collections import defaultdict

from algorithms.spectralclustering import SpectralClusterer
from models.eloConcepts import EloConcepts
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import AvgModel, ItemAvgModel
from utils.data import Data, TimeLimitResponseModificator, ExpDrop, LinearDrop, transform_response_by_time, \
    filter_students_with_many_answers, response_as_binary
from utils.data import compute_corr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import scipy.cluster.hierarchy as hr
import scipy.spatial.distance as dst

from utils.evaluator import Evaluator
from utils.model_comparison import compare_models
from utils.runner import Runner


def get_concepts(data, level=1):
    items = data.get_items_df()
    skills = data.get_skills_df()
    concepts = list(items["skill_lvl_{}".format(level)].unique())
    concepts = filter(lambda x: not np.isnan(x), concepts)
    return {
        skills.loc[int(s), "name"]: list(items[items["skill_lvl_{}".format(level)] == int(s)].index)
        for s in concepts
        }

cache = {}


def compare_model_predictions(data1, data2, model1, model2, plot=True):
    if str(data1) + str(model1) not in cache:
        Evaluator(data1, model1).get_report()
        predictions1 = data1.get_dataframe_test()["prediction"]
        cache[str(data1) + str(model1)] = predictions1
    else:
        predictions1 = cache[str(data1) + str(model1)]

    if str(data2) + str(model2) not in cache:
        Evaluator(data2, model2).get_report()
        predictions2 = data2.get_dataframe_test()["prediction"]
        cache[str(data2) + str(model2)] = predictions2
    else:
        predictions2 = cache[str(data2) + str(model2)]


    Evaluator(data2, model2).get_report()

    # predictions
    predictions = pd.DataFrame(columns=["model1", "model2"])
    predictions["model1"] = predictions1
    predictions["model2"] = predictions2
    predictions_corr = predictions.corr(method="spearman").loc["model1", "model2"]
    if plot:
        plt.plot(predictions["model1"], predictions["model2"], "k.")
        plt.title("Predictions: {}".format(predictions_corr))
        plt.xlabel(str(model1))
        plt.ylabel(str(model2))

    return predictions_corr


def difficulty_stability(datas, models, labels, points, comparable=True, runs=1, eval_data=None):
    df = pd.DataFrame(columns=["data size", "correlation", "models"])
    for i in range(points):
        ratio = (i + 1) / points
        print("Evaluation for {}% of data".format(ratio * 100))

        values = defaultdict(lambda: [])
        for data, model, label in zip(datas, models, labels):
            for run in range(runs):
                d = data(None)
                d.set_seed(run)
                d.set_train_size(ratio)
                d.filter_data(100, 0)
                m = model(None)

                Runner(d, m).run(force=True, only_train=True)

                items = d.get_items()
                if eval_data is None:
                    values[label].append(pd.Series(m.get_difficulties(items), items))
                else:
                    r = Runner(eval_data, m)
                    r.run(force=True, skip_pre_process=True)
                    values[label].append(pd.Series(r._log))

        for i, (data1, model1, label1) in enumerate(zip(datas, models, labels)):
            for data2, model2, label2 in list(zip(datas, models, labels))[i:]:
                print("Computing correlations for " + label1 + " -- " + label2)
                if comparable and label1 != label2:
                    for v1, v2 in zip(values[label1], values[label2]):
                        df.loc[len(df)] = (ratio, v1.corr(v2), label1 + " -- " + label2)
                else:
                    for v1 in values[label1]:
                        for v2 in values[label2]:
                            if v1.sum() == v2.sum() and ratio != 1:
                                continue
                            df.loc[len(df)] = (ratio, v1.corr(v2), label1 + " -- " + label2)

    print(df)
    sns.factorplot(x="data size", y="correlation", hue="models", data=df)


def difficulty_stability2(datas, models, labels, points, runs=1, eval_data=None):
    filename = "../../data/matmat/2016-01-04/tmp.data.pd"
    df = pd.DataFrame(columns=["students", "correlation", "models"])
    student_count = len(datas[0](None).get_students())
    for i in range(points):
        ratio = (i + 1) / points
        print("Evaluation for {}% of data".format(ratio * 100))

        for data, model, label in zip(datas, models, labels):
            for run in range(runs):
                d = data(None)
                d.set_seed(run)
                d.set_train_size(ratio)
                d.filter_data(100, 0)
                d.get_dataframe_train().to_pickle(filename)
                m1 = model(None)
                m2 = model(None)
                d1 = Data(filename, train_size=0.5, train_seed=run + 42)
                d2 = Data(filename, train_size=0.5, train_seed=-run - 42)

                Runner(d1, m1).run(force=True, only_train=True)
                Runner(d2, m2).run(force=True, only_train=True)

                items = d.get_items()
                if eval_data is None:
                    v1 = pd.Series(m1.get_difficulties(items), items)
                    v2 = pd.Series(m2.get_difficulties(items), items)
                else:
                    r1 = Runner(eval_data(None), m1)
                    r2 = Runner(eval_data(None), m2)
                    r1.run(force=True, skip_pre_process=True)
                    r2.run(force=True, skip_pre_process=True)
                    v1 = pd.Series(r1._log)
                    v2 = pd.Series(r2._log)
                df.loc[len(df)] = (ratio * student_count, v1.corr(v2), label)

    print(df)
    sns.factorplot(x="students", y="correlation", hue="models", data=df)


def prediction_quality(datas, models, labels, points, runs=1):
    filename = "../../data/matmat/2016-01-04/tmp2.data.pd"
    df = pd.DataFrame(columns=["~answers", "rmse", "models"])
    data_size = len(datas[0](None).get_dataframe_all())
    for i in range(points):
        ratio = (i + 1) / points
        print("Evaluation for {}% of data".format(ratio * 100))

        for data, model, label in zip(datas, models, labels):
            for run in range(runs):
                d = data(None)
                d.set_seed(run)
                d.set_train_size(ratio)
                d.filter_data(100, 0)
                d.get_dataframe_train().to_pickle(filename)

                d = Data(filename)
                m = model(None)

                Runner(d, m).run(force=True, only_train=True)
                report = Evaluator(d, m).get_report(force_evaluate=True, force_run=True)
                df.loc[len(df)] = (ratio * data_size, report["rmse"], label)

    print(df)
    sns.factorplot(x="~answers", y="rmse", hue="models", data=df)


def compare_more_models(experiments, eval_data, sort_labels=False, runs=1):
    labels = sorted(experiments.keys())

    results = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for label in labels: results[label][label] = 1
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            print(label1, label2)
            for run in range(runs):
                data1 = experiments[label1][0](label1)
                data2 = experiments[label2][0](label2)
                data1.set_seed(run)
                data2.set_seed(run)
                c = compare_model_predictions(data1, data2,
                                  experiments[label1][1](label1), experiments[label2][1](label2), plot=False)
                results[label1][label2] = c
                results[label2][label1] = c

    df = pd.DataFrame(columns=["labels", "rmse"])
    for label in labels:
        r = Evaluator(experiments[label][0](label), experiments[label][1](label)).get_report()
        df.loc[len(df)] = (label, r["rmse"])

    plt.subplot(221)
    plt.title("Correlations of predictions")
    sns.heatmap(results)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.subplot(222)
    compare_models(eval_data, [experiments[label][1](label) for label in labels], answer_filters={
            # "response >7s-0.5": transform_response_by_time(((7, 0.5),)),
            "long (30) students": filter_students_with_many_answers(number_of_answers=30),
        }, runs=runs, hue_order=False)
    plt.subplot(223)
    compare_models([experiments[label][0](label) for label in labels],
                   [experiments[label][1](label) for label in labels],
                   names=labels,
                   metric="rmse", force_evaluate=False, answer_filters={
            "binary": response_as_binary(),
            "response >7s-0.5": transform_response_by_time(((7, 0.5),), binarize_before=True),
        }, runs=runs, hue_order=False)
    plt.subplot(224)
    compare_models([experiments[label][0](label) for label in labels],
                   [experiments[label][1](label) for label in labels],
                   names=labels,
                   metric="AUC", force_evaluate=False, runs=runs, hue_order=False)




model_flat = lambda label: EloPriorCurrentModel(KC=2, KI=0.5)
model_hierarchical = lambda label: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
filename = "../../data/matmat/2016-01-04/answers.pd"
data = lambda l: Data(filename, train_size=0.7)
data_less_items = lambda l: Data(filename.replace(".pd", ".less_items.pd"), train_size=0.7)
data_test = lambda l: Data(filename.replace(".pd", ".test.pd"))
data_train = lambda l: Data(filename.replace(".pd", ".train.pd"))
data_skip_time = lambda l: Data(filename, train_size=0.7, response_modification=TimeLimitResponseModificator([(7, 0.5)]))
data_exp_time = lambda l: Data(filename, train_size=0.7, response_modification=ExpDrop(5, 0.9))
concepts_5 = get_concepts(data(None), level=1)
concepts_10 = get_concepts(data(None), level=2)
concepts_many = get_concepts(data(None), level=3)



if False:
    plt.figure(figsize=(10, 10), dpi=100)
    compare_model_predictions(data(None), data(None), model_flat(None), model_hierarchical(None))
    # print(compare_models(data_time_2, data_time_2b, model_flat(None), model_flat(None)))
    plt.show()

models = {
    "Item avg.": (data, lambda l: ItemAvgModel()),
    "Flat": (data, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "Hierarchical": (data, lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "Concepts 5": (data, lambda l: EloConcepts(concepts=concepts_5)),
    "Concepts 10": (data, lambda l: EloConcepts(concepts=concepts_10)),
    "Item avg. + T": (data_skip_time, lambda l: ItemAvgModel()),
    "Flat + T": (data_skip_time, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "Hierarchical + T": (data_skip_time, lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "Concepts 5 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_5)),
    "Concepts 10 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_10)),
    "Item avg. + expT": (data_exp_time, lambda l: ItemAvgModel()),
    "Flat + expT": (data_exp_time, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "Hierarchical + expT": (data_exp_time, lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "Concepts 5 + expT": (data_exp_time, lambda l: EloConcepts(concepts=concepts_5)),
    "Concepts 10 + expT": (data_exp_time, lambda l: EloConcepts(concepts=concepts_10)),
}

# compare_more_models(models, data(None), runs=10)



models_concepts = {
    "1Flat": (data, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "1Concepts 5": (data, lambda l: EloConcepts(concepts=concepts_5)),
    "1Concepts 10": (data, lambda l: EloConcepts(concepts=concepts_10)),
    "1Concepts many": (data, lambda l: EloConcepts(concepts=concepts_many)),
    "2Flat + T": (data_skip_time, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "2Concepts 5 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_5)),
    "2Concepts 10 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_10)),
    "2Concepts many + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_many)),
}

# compare_more_models(models_concepts, data(None), runs=10)


# print(data.get_dataframe_all()["response_time"].median())

if False:
    # prediction_quality(
    difficulty_stability2(
        # [data_train, data_train, data_train, data_train, data_train],
        [data, data, data_less_items, data, data, data],
        [
            lambda l: ItemAvgModel(),
            model_flat,
            model_flat,
            model_hierarchical,
            lambda l: EloConcepts(concepts=concepts_5),
            lambda l: EloConcepts(concepts=concepts_10)
        ],
        ["Item Avg", "Flat", "Flat less items", "Hierarchical", "Concepts 5", "Concepts 10"],
        10, runs=5,
        # eval_data=data_test
    )

if True:
    difficulty_stability2(
    # prediction_quality(
            [data, data, data, data, data, data, data],
            # [data_train, data_train, data_train, data_train, data_train, data_train, data_train],
            [
                lambda l: EloPriorCurrentModel(),
                lambda l: EloPriorCurrentModel(alpha=0.2),
                lambda l: EloPriorCurrentModel(alpha=5),
                lambda l: EloPriorCurrentModel(beta=0.5),
                lambda l: EloPriorCurrentModel(beta=0.02),
                lambda l: EloPriorCurrentModel(KC=8, KI=2),
                lambda l: EloPriorCurrentModel(KC=2, KI=0.5)
            ],
            ["Default (alpha=1, beta=0.1, K=1,1)", "alpha=0.5", "alpha=2", "beta=0.5", "beta=0.02", "K=8,2", "Fitted (K=2,0.5)"],
            10, runs=2,
    )

if False:
    # prediction_quality(
    difficulty_stability2(
        [
            data_train,
            lambda l: Data(filename.replace(".pd", ".train.pd"), response_modification=TimeLimitResponseModificator([(7, 0.5)])),
            lambda l: Data(filename.replace(".pd", ".train.pd"), response_modification=ExpDrop(5, 0.9))
        ],
        # [data, data_skip_time, data_exp_time],
        [model_flat, model_flat, model_flat,],
        ["Flat", "Flat + T", "Flat + expT"],
        10, runs=10,
        eval_data=data_test
    )


if False:
    data = Data(filename, train_size=0.7)
    data.get_dataframe_train().to_pickle(filename.replace(".pd", ".train.pd"))
    data.get_dataframe_test().to_pickle(filename.replace(".pd", ".test.pd"))

if False:
    d = data(None)
    df = d.get_dataframe_all()
    items = d.get_items_df()

    df.to_pickle(filename.replace(".pd", ".less_items.pd"))

plt.show()
