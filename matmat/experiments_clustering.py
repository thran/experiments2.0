from algorithms.spectralclustering import SpectralClusterer
from utils import data, evaluator, utils
from utils.data import compute_corr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

d = data.Data("../data/matmat/2015-11-20/answers.pd", only_first=True)
# d = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.5)]), only_first=True)
concepts = d.get_concepts()
items = d.get_items_df()
skills = d.get_skills_df()
items = items.join(skills, on="skill")



items = items[items["skill_lvl_1"] == 2]

corr = compute_corr(d)
corr = pd.DataFrame(corr, index=items.index, columns=items.index)
solution = np.zeros(len(corr.index))

sc = SpectralClusterer(corr, kcut=corr.shape[0] / 2 )
labels = sc.run(cluster_number=4, KMiter=50,  sc_type=1)

colors = "rgbyk"
visualizations = list(items["visualization"].unique())

for i, p in enumerate(corr.columns):
    item = items.loc[p]
    # plt.plot(sc.eig_vect[n,1], sc.eig_vect[n,2], "o", color=colors[visualizations.index(item["visualization"])], label=item["visualization"])
    plt.plot(sc.eig_vect[i, 1], sc.eig_vect[i, 2], "o", color=colors[labels[i]])
    plt.text(sc.eig_vect[i, 1], sc.eig_vect[i, 2], item["name"])

plt.legend(loc=0)
plt.show()