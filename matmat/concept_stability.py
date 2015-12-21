from skll.metrics import kappa
from sklearn.metrics import v_measure_score

from matmat.experiments_clustering import item_clustering, concept_clustering
from utils.data import Data

data_filename = "../data/matmat/2015-12-16/answers.pd"

def stability(data_filename, skill, cluster_number=3, method="spectral_concepts", runs=10, sample_size=0.7):
    results = []

    for i in range(runs):
        data = Data(data_filename, train_seed=i, train_size=0.7)
        if method == "spectral_items":
            results.append(item_clustering(data, skill, cluster_number=cluster_number))
        if method == "spectral_concepts":
            results.append(concept_clustering(data, skill, cluster_number=cluster_number))

    print(results)


stability(data_filename, "numbers <= 10", runs=10)

