from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn

from algorithms.spectralclustering import SpectralClusterer

answers = pd.read_csv('../../data/umimecesky/2016-05-18/doplnovackaLog.csv', sep=';')
questions = pd.read_csv('../../data/umimecesky/2016-05-18/doplnovackaZadani.csv', sep=';', index_col='word')
# answers = answers.join(questions, on='word',rsuffix='_question')

question_solutions = {}
for question in questions.itertuples():
    question_solutions[question[0]] = question[4] if question[0].replace('_', question[4]) == question[3] else question[5]
    questions.loc[question[0], 'correct'] = question_solutions[question[0]]

questions.loc[questions['variant1'] == questions['correct'], 'correct_variant'] = 0
questions.loc[questions['variant2'] == questions['correct'], 'correct_variant'] = 1


def filter_users(answers):
    answers = answers.join(questions, on='word',rsuffix='_question')

    bad_users = []

    def user(df):
        variants_1 = len(df[(df['correct'] == 1) ^ (df['correct_variant'] == 1)])
        if variants_1 == 0 or variants_1 == len(df) or (
            len(df) > 10 and (variants_1 <= 1 or variants_1 >= len(df) - 1)):
            # print(df.iloc[0]['user'], variants_1, len(df) - variants_1)
            bad_users.append(df.iloc[0]['user'])

    answers.groupby(['user', 'concept']).apply(user)

    return answers[~answers['user'].isin(bad_users)]


def cluster_concept(answers, concept=None):
    if concept is not None:
        answers = answers[answers['concept'] == concept]

    answers = answers.groupby(['user', 'word']).first().reset_index()

    corr = answers.pivot('user', 'word', 'correct').corr()
    corr[corr < 0] = 0
    sc = SpectralClusterer(corr, kcut=corr.shape[0] / 2, mutual=True)
    # sc = SpectralClusterer(corr, kcut=30, mutual=True)
    labels = sc.run(cluster_number=2, KMiter=50, sc_type=2)

    colors = "rgbyk"
    for i, word in enumerate(corr.columns):
        var = questions.get_value(word, 'correct_variant')
        if type(var) == np.ndarray:
            var = var[0]
        plt.plot(sc.eig_vect[i, 1], sc.eig_vect[i, 2], "o", color=colors[int(var)])
        plt.text(sc.eig_vect[i, 1], sc.eig_vect[i, 2], word)


# answers = filter_users(answers)
cluster_concept(answers, concept=1)
plt.show()
