import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import seaborn


df = pd.read_pickle("../../matmat/model impact/corr-all.pd")
print(df)

model = TSNE(learning_rate=1000, n_iter=100000, init='pca')
result = model.fit_transform(df)

for name, (x, y) in zip(df.columns, result):
    plt.plot(x, y, ".")
    plt.text(x, y, name)
plt.show()
