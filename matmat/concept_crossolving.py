from utils.data import Data
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt

TRASHOLD = 25

d = Data("../data/matmat/2015-11-20/answers.pd")
answers = d.get_dataframe_all()
items = d.get_items_df()
skills = d.get_skills_df()
items = items.join(skills, on="skill_lvl_1")

concepts = items["name"].unique()
sts = {}

for concept in concepts:
    print(concept)
    its = list(items[items["name"] == concept].index)
    students = answers[answers["item"].isin(its)].groupby("student").size()
    students = students[students >= TRASHOLD]
    print(len(students))
    sts[concept] = students

data = pd.DataFrame(index=concepts, columns=concepts, dtype=float)

for concept1 in concepts:
    for concept2 in concepts:
        count = len(set(sts[concept1]) & set(sts[concept2]))
        print(concept1, concept2, count)
        data[concept1][concept2] = count

print(data)
plt.switch_backend('agg')
sns.heatmap(data, annot=True)
plt.show()

plt.title("Student with {} answer in concept".format(TRASHOLD))
plt.savefig("concepts crossolving - {}.png".format(TRASHOLD))
