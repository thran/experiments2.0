from collections import defaultdict

import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.eloConcepts import EloConcepts
from models.eloPriorCurrent import EloPriorCurrentModel
from models.time_models import TimePriorCurrentModel
from time_impact.time_utils import get_difficulties, rolling_success
from utils import data as d
from utils.data import LinearDrop, items_in_concept, \
    TimeLimitResponseModificator
from utils.runner import Runner

basic_data = lambda response_modification: d.Data("../data/matmat/2016-12-11/answers.pd",
                                                  response_modification=response_modification,
                                                  filter_items={'visualization': ['written_question']},
                                                  )
concepts = basic_data(None).get_concepts()
# basic_model = lambda time_tra: EloPriorCurrentModel(KC=2, KI=0.5)
basic_model = lambda time_tra: EloConcepts(concepts=concepts, separate=True)
time_median = 7
min_answers = 20
item_names = pd.read_csv('../data/matmat/2016-12-11/items.csv', index_col='id')['question']


results = defaultdict(lambda: defaultdict(lambda : {}))
for response_modification in [
    None,
    # TimeLimitResponseModificator([(time_median, 0.5)]),
    LinearDrop(time_median * 2),
]:
    data = basic_data(response_modification)
    model = basic_model(None)
    data.trim_times()
    answers = data.get_dataframe_all()

    Runner(data, model).run(force=True)

    for concept in ['numbers', 'addition', 'subtraction', 'multiplication', 'division']:
        items = list(set(data.get_items()) & set(items_in_concept(data, concept)))
        concept_answers = answers[answers['item'].isin(items)]
        students = concept_answers.groupby('student').apply(len)
        students = students[students >= min_answers]

        results[concept][str(response_modification)] = (
            items,
            model.get_skills(students.index, concept),
            model.get_difficulties(items),
            concept_answers.groupby('student')['response_time'].median().loc[students.index],
            concept_answers.groupby('student')['correct'].mean().loc[students.index],
            concept_answers.groupby('item')['response_time'].median().loc[items],
            concept_answers.groupby('item')['correct'].mean().loc[items],
        )

for concept, r in results.items():
    df_difficulties = pd.DataFrame()
    df_skills = pd.DataFrame()
    for response_modification, (_, skills, difficulties, _, _, _, _) in sorted(r.items()):
        # print(concept, response_modification, len(skills), len(difficulties))

        df_difficulties[response_modification] = difficulties
        df_skills[response_modification] = skills

    corr_s, count_s = df_skills.corr(method='spearman').ix[0, 1], len(list(r.values())[0][1]),
    corr_d, count_d = df_difficulties.corr(method='spearman').ix[0, 1], len(list(r.values())[0][2]),

    plt.figure(figsize=(30, 10))
    plt.suptitle(concept)
    if 1: # plt
        plt.subplot(1, 2, 1)
        plt.title('{:.3f} ({:>4})'.format(corr_s, count_s))
        plt.xlabel(df_skills.columns[0])
        plt.ylabel(df_skills.columns[1])
        _, _, _, rts, cs, _, _ = results[concept][str(None)]
        plt.scatter(
            df_skills.ix[:, 0],
            df_skills.ix[:, 1],
            s = rts / rts.max() * 500,
            alpha=0.5,
            cmap='viridis',
            c = cs,
            vmin=0.
        )
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title('{:.3f} ({:>4})'.format(corr_d, count_d))
        plt.xlabel(df_skills.columns[0])
        plt.ylabel(df_skills.columns[1])
        items_ids, _, _, _, _, rts, cs = results[concept][str(None)]
        plt.scatter(
            df_difficulties.ix[:, 0],
            df_difficulties.ix[:, 1],
            s = rts / rts.max() * 500,
            alpha=0.5,
            cmap='viridis',
            c = cs,
            vmin=cs.min(),
            vmax=cs.max(),
        )
        plt.colorbar()
        if 1:
            for item, x, y in zip(items_ids, df_difficulties.ix[:, 0], df_difficulties.ix[:, 1]):
                plt.text(x, y, item_names.loc[item], alpha=0.5, fontsize=10)

    print('{:<15} -- skill sp: {:.3f} ({:>4}), difficulty sp: {:.3f} ({:>4})'.format(
        concept,
        corr_s, count_s,
        corr_d, count_d,
    ))

plt.show()
