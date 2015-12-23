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

from utils.runner import Runner


def compare_models(data1, data2, model1, model2):
    runner1 = Runner(data1, model1)
    runner2 = Runner(data2, model2)
    runner1.run(force=True)
    runner2.run(force=True)

    # difficulties
    items = list(set(data1.get_items()) & set(data2.get_items()))
    difficulties = pd.DataFrame(columns=["model1", "model2"], index=items)
    difficulties["model1"] = model1.get_difficulties(items)
    difficulties["model2"] = model2.get_difficulties(items)
    plt.subplot(221)
    plt.plot(difficulties["model1"], difficulties["model2"], "k.")
    plt.title("Difficulties: {}".format(difficulties.corr().loc["model1", "model2"]))
    plt.xlabel(str(model1))
    plt.ylabel(str(model2))

    # skills
    students = list(set(data1.get_students()) & set(data2.get_students()))
    skills = pd.DataFrame(index=students, columns=["model1", "model2"])
    skills["model1"] = model1.get_skills(students)
    skills["model2"] = model2.get_skills(students)
    plt.subplot(222)
    plt.plot(skills["model1"], skills["model2"], "k.")
    plt.title("Skills: {}".format(skills.corr().loc["model1", "model2"]))
    plt.xlabel(str(model1))
    plt.ylabel(str(model2))

    # predictions
    predictions = pd.DataFrame(index=students, columns=["model1", "model2"])
    predictions["model1"] = pd.Series(runner1._log)
    predictions["model2"] = pd.Series(runner2._log)
    plt.subplot(223)
    plt.plot(predictions["model1"], predictions["model2"], "k.")
    plt.title("Predictions: {}".format(predictions.corr().loc["model1", "model2"]))
    plt.xlabel(str(model1))
    plt.ylabel(str(model2))



data = Data("../data/matmat/2015-12-16/answers.pd")
data_time_2 = Data("../data/matmat/2015-12-16/answers.pd", response_modification=TimeLimitResponseModificator([(5, 0.5)]))
plt.figure(figsize=(10, 10), dpi=100)

model_flat = lambda: EloPriorCurrentModel(KC=2, KI=0.5)
model_hierarchical = lambda: EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)


# compare_models(data, data, model_flat(), model_hierarchical())
compare_models(data, data_time_2, model_flat(), model_flat())
plt.show()