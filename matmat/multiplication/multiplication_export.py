from matmat.experiments_clustering import all_in_one
from models.eloPriorCurrent import EloPriorCurrentModel
from utils import utils
from utils.data import Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from utils.evaluator import Evaluator

data = Data("../../data/matmat/2016-06-27/answers.pd", train_size=1)
data.trim_times()
answers = data.get_dataframe_all()

def grid(data, model):
    utils.grid_search(data, model,
          {"KC": 3, "KI": 0.5}, {
          # {"alpha": 0.25, "beta": 0.02}, {
              "alpha": np.arange(0.4, 1.7, 0.2),
              "beta": np.arange(0., 0.2, 0.02),
              # "KC": np.arange(1.5, 5.0, 0.25),
              # "KI": np.arange(0, 2.5, 0.25),
          # }, plot_axes=["KC", "KI"])
        }, plot_axes=["alpha", "beta"])

    plt.show()


items = data.get_items_df()
items = items[(items["skill_lvl_2"] == 210) & ~items["skill_lvl_3"].isnull()].loc[:, ("question", "answer", "visualization")]
items = items[items["visualization"] == "free_answer"]

answers[answers["item"].isin(items.index)].to_pickle("../../data/matmat/2016-01-04/answers-multiplication.pd")
data_multiplication = Data("../../data/matmat/2016-01-04/answers-multiplication.pd")
model = EloPriorCurrentModel(alpha=1.4, beta=0.1, KC=3, KI=0.5)

items = items.join(pd.Series(answers.groupby("item").size(), name="answer_count"))
items = items.join(pd.Series(answers.groupby("item").apply(lambda i: i["correct"].sum() / len(i)), name="success_rate"))
items = items.join(pd.Series(answers.groupby("item")["response_time"].median(), name="response_time"))

Evaluator(data_multiplication, model).get_report(force_run=True)
items["model_difficulty"] = model.get_difficulties(items.index)
items["model_difficulty"] -= items["model_difficulty"].mean()

skills = items.groupby("question").agg({
    "answer_count": "sum",
    "success_rate": "mean",
    "response_time": "mean",
    "model_difficulty": "mean",
})

dfSR = pd.DataFrame(index=range(1, 11)[::-1], columns=range(1, 11), dtype=float)
dfD = pd.DataFrame(index=range(1, 11)[::-1], columns=range(1, 11), dtype=float)
dfAC = pd.DataFrame(index=range(1, 11)[::-1], columns=range(1, 11), dtype=float)
dfRT = pd.DataFrame(index=range(1, 11)[::-1], columns=range(1, 11), dtype=float)
for q, skill in skills.iterrows():
    a,b = map(int, q.split("x"))
    dfSR.loc[a, b] = skill["success_rate"]
    dfD.loc[a, b] = skill["model_difficulty"]
    dfAC.loc[a, b] = skill["answer_count"]
    dfRT.loc[a, b] = skill["response_time"]
plt.switch_backend('agg')
plt.figure(figsize=(20, 16))
plt.subplot(221)
plt.title("Success rate")
sns.heatmap(dfSR, annot=True)
plt.subplot(222)
plt.title("Model difficulty")
sns.heatmap(dfD, annot=True)
plt.subplot(223)
plt.title("Answer count")
sns.heatmap(dfAC, annot=True, fmt=".0f")
plt.subplot(224)
plt.title("Median od response times")
sns.heatmap(dfRT, annot=True, fmt=".1f")
# plt.show()
plt.savefig("multiplication - free_answer.png")


print(skills)
skills[["answer_count", "success_rate", "model_difficulty", "response_time"]].to_csv("multiplication.csv")
# print(items[items["question"] == "7x"])

plt.figure()
all_in_one(data, "multiplication1", 3)
# plt.show()