from utils.data import convert_slepemapy, Data
import pandas as pd
import matplotlib.pylab as plt

df = convert_slepemapy("../data/slepemapy/2016-ab-target-difficulty/answers.csv")
# data = Data("../data/slepemapy/2016-ab-target-difficulty/answers.csv")


def drop_off(data):
    counts = data.groupby("student").size()
    r = range(1, 100)
    return r, [sum(counts.values >= count) / len(counts) for count in r]


print(df.groupby("experiment_setup_id")["correct"].mean())
for setup in df["experiment_setup_id"].unique():
    x, y = drop_off(df[df["experiment_setup_id"] == setup])
    plt.plot(x, y, label=setup)
plt.legend(loc=2)
plt.show()