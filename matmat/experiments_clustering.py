from algorithms.spectralclustering import SpectralClusterer
from utils import data, evaluator, utils
from utils.data import compute_corr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

d = data.Data("../data/matmat/2015-11-20/answers.pd", only_first=True)
dt = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.5)]), only_first=True)


def item_clustering(data, skill):
    pk, level = data.get_skill_id(skill)
    items = data.get_items_df()
    items = items[items["skill_lvl_" + str(level)] == pk]
    items = items[items["visualization"] != "pairing"]

    corr = compute_corr(data)
    corr = pd.DataFrame(corr, index=items.index, columns=items.index)

    print("Corr ({}) contain total {} values and from that {} nans".format(corr.shape, corr.size, corr.isnull().sum().sum()))
    corr[corr.isnull()] = 0

    sc = SpectralClusterer(corr, kcut=corr.shape[0] / 2, mutual=True)
    # sc = SpectralClusterer(corr, kcut=30, mutual=True)
    labels = sc.run(cluster_number=3, KMiter=50,  sc_type=2)

    colors = "rgbyk"
    visualizations = list(items["visualization"].unique())

    for i, p in enumerate(corr.columns):
        item = items.loc[p]
        plt.plot(sc.eig_vect[i,1], sc.eig_vect[i,2], "o", color=colors[visualizations.index(item["visualization"])])
        # plt.plot(sc.eig_vect[i, 1], sc.eig_vect[i, 2], "o", color=colors[labels[i]])
        plt.text(sc.eig_vect[i, 1], sc.eig_vect[i, 2], item["name"])

    for i, vis in enumerate(visualizations):
        plt.plot(0, 0, "o", color=colors[i], label=vis)
    plt.title(data)

    plt.legend(loc=3)
    plt.show()


def concept_clustering(data, skill):
    pk, level = data.get_skill_id(skill)
    items = data.get_items_df()
    items = items[items["skill_lvl_" + str(level)] == pk]
    skills = data.get_skills_df()
    skill_ids = items[~items["skill_lvl_3"].isnull()]["skill_lvl_3"].unique()

    corr = compute_corr(data, merge_skills=True)
    corr = pd.DataFrame(corr, index=skill_ids, columns=skill_ids)
    print("Corr ({}) contain total {} values and from that {} nans".format(corr.shape, corr.size, corr.isnull().sum().sum()))
    corr[corr.isnull()] = 0

    sc = SpectralClusterer(corr, kcut=corr.shape[0] * 0.5, mutual=True)
    labels = sc.run(cluster_number=3, KMiter=50,  sc_type=2)

    colors = "rgbyk"
    for i, p in enumerate(corr.columns):
        skill = skills.loc[int(p)]
        plt.plot(sc.eig_vect[i, 1], sc.eig_vect[i, 2], "o", color=colors[labels[i]])
        plt.text(sc.eig_vect[i, 1], sc.eig_vect[i, 2], skill["name"])

    plt.title(data)
    plt.show()

# item_clustering(d, "numbers <= 20")

concept_clustering(d, "division")