from algorithms.spectralclustering import SpectralClusterer
from matmat.experiments_difficulties import difficulty_vs_time
from utils import data, evaluator, utils
from utils.data import compute_corr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import scipy.cluster.hierarchy as hr
import scipy.spatial.distance as dst

d = data.Data("../data/matmat/2015-12-16/answers.pd", only_first=True)
dt = data.Data("../data/matmat/2015-12-16/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.5)]), only_first=True)


def item_clustering(data, skill, cluster_number=3):
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
    labels = sc.run(cluster_number=cluster_number, KMiter=50, sc_type=2)

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

    return labels


def concept_clustering(data, skill, cluster_number=3):
    pk, level = data.get_skill_id(skill)
    items = data.get_items_df()
    items = items[items["skill_lvl_" + str(level)] == pk]
    skills = data.get_skills_df()
    skill_ids = items[~items["skill_lvl_3"].isnull()]["skill_lvl_3"].unique()

    corr = compute_corr(data, merge_skills=True)
    corr = pd.DataFrame(corr, index=skill_ids, columns=skill_ids)
    print("Corr ({}) contain total {} values and from that {} nans".format(corr.shape, corr.size, corr.isnull().sum().sum()))
    corr[corr.isnull()] = 0

    try:
        sc = SpectralClusterer(corr, kcut=corr.shape[0] * 0.5, mutual=True)
        labels = sc.run(cluster_number=cluster_number, KMiter=50, sc_type=2)
    except np.linalg.linalg.LinAlgError:
        sc = SpectralClusterer(corr, kcut=corr.shape[0] * 0.5, mutual=False)
        labels = sc.run(cluster_number=cluster_number, KMiter=50, sc_type=2)

    colors = "rgbyk"
    for i, p in enumerate(corr.columns):
        skill = skills.loc[int(p)]
        plt.plot(sc.eig_vect[i, 1], sc.eig_vect[i, 2], "o", color=colors[labels[i]])
        plt.text(sc.eig_vect[i, 1], sc.eig_vect[i, 2], skill["name"])

    plt.title(data)
    return labels

# item_clustering(d, "numbers <= 20")

# concept_clustering(d, "division")


def hierarchical_clustering(data, skill,  method='single', metric='euclidean', dendrogram=True, concepts=False, corr_as_vectors=False):
    pk, level = data.get_skill_id(skill)
    items = data.get_items_df()
    skills = data.get_skills_df()
    corr = compute_corr(data, merge_skills=concepts)
    print("Corr ({}) contain total {} values and from that {} nans".format(corr.shape, corr.size, corr.isnull().sum().sum()))
    corr[corr.isnull()] = 0

    if concepts:
        items = items[items["skill_lvl_" + str(level)] == pk]
        skill_ids = items[~items["skill_lvl_3"].isnull()]["skill_lvl_3"].unique()
        corr = pd.DataFrame(corr, index=skill_ids, columns=skill_ids)
        labels = list(skills.loc[corr.index]["name"])

    else:
        items = items[items["skill_lvl_" + str(level)] == pk]
        items = items[items["visualization"] != "pairing"]
        corr = pd.DataFrame(corr, index=items.index, columns=items.index)
        labels = ["{1} - {0}".format(item["name"], item["visualization"][0]) for id, item in list(items.iterrows())]

    if corr_as_vectors:
        Z = hr.linkage(corr, method=method, metric=metric)
    else:
        Z = hr.linkage(dst.squareform(1 - corr), method=method)
    if dendrogram:
        plt.title('{}: method: {}, metric: {}, as vectors: {}'.format(skill, method, metric, corr_as_vectors))
        plt.xlabel('items' if not concepts else "concepts")
        plt.ylabel('distance')
        hr.dendrogram(Z, leaf_rotation=90., leaf_font_size=10., labels=labels)


def all_in_one(data, skill, concepts):
    plt.subplot(221)
    hierarchical_clustering(data, skill, concepts=concepts, corr_as_vectors=False, method="complete")
    plt.subplot(222)
    hierarchical_clustering(data, skill, concepts=concepts, corr_as_vectors=True, method="ward")
    plt.subplot(223)
    if concepts:
        concept_clustering(data, skill)
    else:
        item_clustering(data, skill)
    plt.subplot(224)
    difficulty_vs_time(data, skill, concepts=concepts)




# hierarchical_clustering(d, "numbers <= 10", concepts=False, corr_as_vectors=False, method="average")
# hierarchical_clustering(d, "numbers <= 20", concepts=False, corr_as_vectors=True, method="ward")
# hierarchical_clustering(d, "numbers")

if 0:
    skills = ["numbers", "numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1", "multiplication2", "division1"]
    for skill in skills:
        for i, dat in enumerate([d, dt]):
            print (skill, dat)
            plt.figure(figsize=(25, 15), dpi=150)
            plt.title(skill)
            all_in_one(dat, skill, concepts=True)
            plt.savefig("results/concepts/all_in_one/{}{}.png".format(skill, " - time" if i else ""))

if 0:
    skills = ["numbers", "numbers <= 10", "numbers <= 20"]
    for skill in skills:
        for i, dat in enumerate([d, dt]):
            print (skill, dat)
            plt.figure(figsize=(25, 15), dpi=150)
            plt.title(skill)
            all_in_one(dat, skill, concepts=False)
            plt.savefig("results/concepts/all_in_one/items - {}{}.png".format(skill, " - time" if i else ""))

# all_in_one(d, "numbers", concepts=False)
# all_in_one(dt, "numbers", concepts=True)
# all_in_one(d, "subtraction <= 10", concepts=True)
# plt.show()