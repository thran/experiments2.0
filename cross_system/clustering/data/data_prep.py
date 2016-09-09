import math
import random

import numpy as np
import pandas as pd


def matmat():
    answers = pd.read_pickle('../../../data/matmat/2016-06-27/answers.pd')
    answers = answers.groupby(['student', 'item']).first().reset_index()
    items = pd.read_csv('../../../data/matmat/2016-06-27/items.csv', index_col='id')
    items = items[items['visualization'] != 'pairing']
    items = items[items['skill_lvl_1'] == 2]

    answers = answers[answers['item'].isin(items.index)]
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]

    items = items.loc[:, ['question', 'visualization']].rename(columns={'question': 'name', 'visualization': 'concept'})

    answers.to_pickle('matmat-numbers-answers.pd')
    items.to_pickle('matmat-numbers-items.pd')
    # print(items)


def math_garden():
    concept = 'subtraction'
    answers = pd.read_pickle('../../../data/mathgarden/{}.pd'.format(concept))
    answers = answers.rename(columns={'user_id': 'student', 'item_id': 'item', 'correct_answered': 'correct', 'response_in_milliseconds': 'response_time'})
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]
    answers = answers.groupby(['student', 'item']).first().reset_index()
    answers['student'] = answers['student'].astype(int)
    answers['item'] = answers['item'].astype(int)
    answers['response_time'] /= 1000

    item_ids = answers['item'].unique()
    items = pd.DataFrame(np.array([item_ids, [concept] * len(item_ids)]).T, index=item_ids, columns=['name', 'concept'])

    answers.to_pickle('math_garden-{}-answers.pd'.format(concept))
    items.to_pickle('math_garden-{}-items.pd'.format(concept))


def cestina():
    answers = pd.read_csv('../../../data/umimecesky/2016-05-18/doplnovackaLog.csv', sep=';')
    questions = pd.read_csv('../../../data/umimecesky/2016-05-18/doplnovackaZadani.csv', sep=';', index_col='word')
    answers = answers.join(questions, on='word',rsuffix='_question')

    question_solutions = {}
    for question in questions.itertuples():
        question_solutions[question[0]] = question[4] if question[0].replace('_', question[4]) == question[3] else question[5]
        questions.loc[question[0], 'correct'] = question_solutions[question[0]]

    questions.loc[questions['variant1'] == questions['correct'], 'correct_variant'] = 0
    questions.loc[questions['variant2'] == questions['correct'], 'correct_variant'] = 1
    questions = questions.set_index(questions['id'])

    questions = questions[questions['concept'] == 7]

    items = questions.loc[:, ['solved', 'correct']].rename({'solved': 'name', 'correct': 'concept'})

    answers = answers.rename(columns={'id': 'item', 'user': 'student'})
    answers = answers[answers['item'].isin(items.index)]
    answers['response_time'] = 0
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]

    print(items)
    answers.to_pickle('cestina-Z-answers.pd')
    items.to_pickle('cestina-Z-items.pd')


def math_garden_all():
    concepts = ['addition', 'subtraction', 'multiplication']

    answers = pd.concat([pd.read_pickle('math_garden-{}-answers.pd'.format(c)) for c in concepts])
    answers.to_pickle('math_garden-all-answers.pd')
    items = pd.concat([pd.read_pickle('math_garden-{}-items.pd'.format(c)) for c in concepts])
    items.to_pickle('math_garden-all-items.pd')
    
    
def simulated(n_students=100, n_concepts=5, n_items=20):
    def sigmoid(x, c=0):
        return c + (1 - c) / (1 + math.exp(-x))

    skill = np.random.randn(n_students, n_concepts)
    items = np.array([[i,  i // n_items] for i in range(n_concepts * n_items)])
    difficulty = np.random.randn(len(items)) - 0.5 # shift to change overall difficulty

    answers = []
    for s in range(n_students):
        for i, concept in items:
            prob = sigmoid(skill[s, concept] - difficulty[i])
            answers.append([s, i, 0, 1 *(random.random() < prob) ])

    answers = pd.DataFrame(answers, columns=['student', 'item', 'response_time', 'correct'])
    items = pd.DataFrame(items, columns=['name', 'concept'])

    answers.to_pickle('simulated-s{}-c{}-i{}-answers.pd'.format(n_students, n_concepts, n_items))
    items.to_pickle('simulated-s{}-c{}-i{}-items.pd'.format(n_students, n_concepts, n_items))


# math_garden()
# matmat()
# math_garden_all()
# cestina()

simulated()
