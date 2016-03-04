import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import seaborn

answers = pd.read_pickle("../../data/slepemapy/2016-ab-target-difficulty/answers.pd")
flashcards =pd.read_csv("../../data/slepemapy/2016-ab-target-difficulty/flashcards.csv", index_col="item_id")
answers = answers.join(flashcards, on="item")


def filter_flashcards(answers, context, term):
    return answers[(answers["context_name"] == context) & (answers["term_type"] == term)]

answers = filter_flashcards(answers, "Europe", "state")
answers = answers.drop_duplicates(['student', 'item'])

# answers["value"] = answers["correct"] * 1
answers["value"] = answers["correct"] - answers["guess"]
answers.loc[answers["value"] < 0, "value"] = 0

df = answers.pivot(index='student', columns="term_name", values="value").corr()
df[df < df.mean()] = 0
print(df)

model = TSNE(learning_rate=1000, n_iter=100000, init='pca')
result = model.fit_transform(df)

for name, (x, y) in zip(df.columns, result):
    plt.plot(x, y, ".")
    plt.text(x, y, name)
plt.show()



