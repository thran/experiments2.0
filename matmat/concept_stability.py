from itertools import combinations
from skll.metrics import kappa
from sklearn.metrics import v_measure_score
from matmat.experiments_clustering import item_clustering, concept_clustering, hierarchical_clustering
from utils.data import Data
import seaborn as sns
import matplotlib.pylab as plt

data_filename = "../data/matmat/2015-12-16/answers.pd"

def stability(data_filename, skill, cluster_number=3, method="hierarchical_concepts", runs=10, sample_size=0.7):
    results = []

    for i in range(runs):
        data = Data(data_filename, train_seed=i, train_size=sample_size)
        if method == "spectral_items":
            data.only_first()
            results.append(item_clustering(data, skill, cluster_number=cluster_number, plot=False))
        if method == "spectral_concepts":
            results.append(concept_clustering(data, skill, cluster_number=cluster_number, plot=False))
        if method == "hierarchical_concepts":
            results.append(hierarchical_clustering(data, skill, concepts=True, cluster_number=cluster_number,  corr_as_vectors=False, method="complete", dendrogram=False))
        if method == "hierarchical_concepts_vectors":
            results.append(hierarchical_clustering(data, skill, concepts=True, cluster_number=cluster_number,  corr_as_vectors=True, method="ward", dendrogram=False))

    values = []
    for l1, l2 in combinations(results, 2):
        values.append(v_measure_score(l1, l2))

    plt.title("{} - {} ".format(method, skill))
    plt.hist(values)

if 1:
    skills = ["numbers", "numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1", "multiplication2", "division1"]
    method = "hierarchical_concepts_vectors"
    for skill in skills:
        plt.figure()
        stability(data_filename, skill, method=method, runs=20, sample_size=0.7)
        plt.savefig("results/concepts/stability/{}-{}.png".format(method, skill))


# plt.show()
