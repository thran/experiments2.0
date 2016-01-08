from algorithms.spectralclustering import SpectralClusterer
from models.eloConcepts import EloConcepts
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import AvgModel, ItemAvgModel
from utils.data import Data, TimeLimitResponseModificator, ExpDrop, LinearDrop, transform_response_by_time
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
    print(len(cache))
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


def compare_more_models(experiments, eval_data, sort_labels=False):
    labels = sorted(experiments.keys())

    results = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for label1 in labels:
        for label2 in labels:
            c = compare_model_predictions(experiments[label1][0](label1), experiments[label2][0](label2),
                                     experiments[label1][1](label1), experiments[label2][1](label2), plot=False)
            results[label1][label2] = c

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
        "response >7s-0.5": transform_response_by_time(((7, 0.5),))
    })
    plt.subplot(223)
    compare_models([experiments[label][0](label) for label in labels],
                   [experiments[label][1](label) for label in labels],
                   names=labels,
                   metric="rmse", force_evaluate=False)
    plt.subplot(224)
    compare_models([experiments[label][0](label) for label in labels],
                   [experiments[label][1](label) for label in labels],
                   names=labels,
                   metric="AUC", force_evaluate=False)




model_flat = lambda label: EloPriorCurrentModel(KC=2, KI=0.5)
model_hierarchical = lambda label: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
filename = "../../data/matmat/2016-01-04/answers.pd"
data = lambda l: Data(filename, train_size=0.7)
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
    "Concepts many": (data, lambda l: EloConcepts(concepts=concepts_many)),
    "Item avg. + T": (data_skip_time, lambda l: ItemAvgModel()),
    "Flat + T": (data_skip_time, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "Hierarchical + T": (data_skip_time, lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "Concepts 5 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_5)),
    "Concepts 10 + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_10)),
    "Concepts many + T": (data_skip_time, lambda l: EloConcepts(concepts=concepts_many)),
    "Item avg. + expT": (data_exp_time, lambda l: ItemAvgModel()),
    "Flat + expT": (data_exp_time, lambda l: EloPriorCurrentModel(KC=2, KI=0.5)),
    "Hierarchical + expT": (data_exp_time, lambda l: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    "Concepts 5 + expT": (data_exp_time, lambda l: EloConcepts(concepts=concepts_5)),
    "Concepts 10 + expT": (data_exp_time, lambda l: EloConcepts(concepts=concepts_10)),
    "Concepts many + expT": (data_exp_time, lambda l: EloConcepts(concepts=concepts_many)),
}

compare_more_models(models, data(None))



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

# compare_more_models(models_concepts, data(None))


# print(data.get_dataframe_all()["response_time"].median())

plt.show()