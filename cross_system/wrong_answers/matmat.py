from cross_system.wrong_answers.statistics import filter_answers
from utils.data import Data
import pandas as pd

data = Data("../../data/matmat/2016-01-04/answers.pd")
# concepts = ["numbers <= 10", "numbers <= 20", "addition <= 10", "subtraction <= 10", "multiplication1"]
concepts = ["numbers", "addition", "subtraction", "multiplication", "division"]

# answers = filter_answers(data, "multiplication1")
# print(wrong.groupby(["answer_expected", "answer"])["id"].count())


def repetition(data):
    answers = data.get_dataframe_all()
    answers = answers.join(data.get_items_df()["question"], on="item")
    answers["timestamp"] = pd.to_datetime(answers["timestamp"])
    wrong = answers[(answers["correct"] != 1) & (~answers["answer"].isnull())]

    for (item, student, answer), ans in wrong.groupby(["question", "student", "answer"]):
        if len(ans) > 1:
            print(ans.loc[ans.index[0], "question"],"=", int(ans.loc[ans.index[0], "answer_expected"]))
            ts = ans.loc[ans.index[0], "timestamp"]
            for i, a in ans.iterrows():
                print(a["answer"], "+" + str(a["timestamp"] - ts))
            print()



repetition(data)