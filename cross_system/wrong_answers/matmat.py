from collections import defaultdict

import re

from cross_system.wrong_answers.statistics import filter_answers
from utils.data import Data
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

data = Data("../../data/matmat/2017-03-29/answers.pd")
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


def type_of_mistakes(data):
    concepts = ["numbers", "addition"]
    concepts = ["numbers", "addition", "subtraction", "multiplication", "division"]
    df = pd.DataFrame(columns=["concept", "mistake_type", "ratio"])
    for concept in concepts:
        print(concept)
        answers = filter_answers(data, concept)
        answers = answers.join(data.get_items_df()["question"], on="item")
        wrong = answers[(answers["correct"] != 1)]
        counts = defaultdict(lambda: 0)
        for (question, answer), count in wrong.groupby(["question", "answer"]).size().iteritems():
            g = re.match(r'(\d+)(.)(\d+)', question)
            if not g:
                correct_answer = int(question)
            else:
                if g.group(2) == "+":
                    op = lambda a, b: a + b
                elif g.group(2) == "x":
                    op = lambda a, b: a * b
                elif g.group(2) == "/":
                    op = lambda a, b: a / b
                elif g.group(2) == "-":
                    op = lambda a, b: a - b
                correct_answer = int(op(int(g.group(1)), int(g.group(3))))

            if pd.isnull(answer):
                counts["missing"] += count
            elif not answer.isdigit():
                counts["not_number"] += count
            elif int(answer) in [correct_answer + 1, correct_answer - 1]:
                counts["answer_plus_minus_1"] += count
            elif concept is not "numbers" and int(answer) in [
                op(int(g.group(1)) + 1, int(g.group(3))),
                op(int(g.group(1)) - 1, int(g.group(3))),
                op(int(g.group(1)), int(g.group(3)) + 1),
                op(int(g.group(1)), int(g.group(3)) - 1) if concept != "division" or int(g.group(3)) != 1 else None
            ]:
                counts["operand_plus_minus_1"] += count
            elif answer.startswith(str(correct_answer)):
                counts["prefix"] += count
            elif len(answer) > 2:
                counts["3_digits_and_more"] += count
            else:
                counts["other"] += count
            counts["all"] += count

        for mistake_type, count in counts.items():
            if mistake_type != "all":
                df.loc[len(df)] = concept, mistake_type, count / counts["all"]

    print(df)
    # sns.barplot(data=df, x="mistake_type", y="ratio", hue="concept")
    sns.barplot(data=df, x="concept", y="ratio", hue="mistake_type")


# repetition(data)

type_of_mistakes(data)
plt.show()
