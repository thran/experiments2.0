import math
import random

import numpy as np
import pandas as pd
import os


def matmat():
    answers = pd.read_pickle('../../../data/matmat/2016-06-27/answers.pd')
    answers = answers.groupby(['student', 'item']).first().reset_index()
    items = pd.read_csv('../../../data/matmat/2016-06-27/items.csv', index_col='id')
    items = items[items['visualization'] != 'pairing']
    print(items['skill_lvl_1'].unique())
    items = items[items['skill_lvl_2'] == 210]

    answers = answers[answers['item'].isin(items.index)]
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]

    items = items.loc[:, ['question', 'visualization']].rename(columns={'question': 'name', 'visualization': 'concept'})

    answers.to_pickle('matmat-multiplication-answers.pd')
    items.to_pickle('matmat-multiplication-items.pd')


def matmat_all():
    concepts = ['numbers', 'addition', 'subtraction', 'multiplication']

    answers = pd.concat([pd.read_pickle('matmat-{}-answers.pd'.format(c)) for c in concepts])
    answers = answers.set_index(np.arange(len(answers)))
    answers.to_pickle('matmat-all-answers.pd')
    i = []
    for c in concepts:
        s = pd.read_pickle('matmat-{}-items.pd'.format(c))
        s['concept'] = c
        i.append(s)
    items = pd.concat(i)
    print(items)
    items.to_pickle('matmat-all-items.pd')

def math_garden(concept='subtraction'):
    answers = pd.read_pickle('../../../data/mathgarden/{}.pd'.format(concept))
    items = pd.read_pickle('../../../data/mathgarden/{}.pd'.format('items'))
    answers = answers.rename(columns={'user_id': 'student', 'item_id': 'item', 'correct_answered': 'correct', 'response_in_milliseconds': 'response_time'})
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]
    answers = answers.groupby(['student', 'item']).first().reset_index()
    answers['student'] = answers['student'].astype(int)
    answers['item'] = answers['item'].astype(int)
    answers['response_time'] /= 1000

    print(items)
    item_ids = answers['item'].unique()
    items = pd.DataFrame(np.array([items.loc[item_ids], [concept] * len(item_ids)]).T, index=item_ids, columns=['name', 'concept'])
    print(item_ids)
    print(items)

    answers.to_pickle('math_garden-{}-answers.pd'.format(concept))
    items.to_pickle('math_garden-{}-items.pd'.format(concept))


def cestina(concept_id=1, concept_name='B'):
    answers = pd.read_csv('../../../data/umimecesky/2016-05-18/doplnovackaLog.csv', sep=';')
    questions = pd.read_csv('../../../data/umimecesky/2016-05-18/doplnovackaZadani.csv', sep=';', index_col='word')

    answers = answers.join(questions, on='word',rsuffix='_question')
    if os.path.exists('../../../data/umimecesky/shluky-{}.csv'.format(concept_name.lower())):
        concepts = pd.read_csv('../../../data/umimecesky/shluky-{}.csv'.format(concept_name.lower()))
        concepts = pd.melt(concepts)
        concepts = concepts.loc[concepts['value'] == concepts['value'] , :]
        concepts = pd.Series(data=list(concepts['variable']), index=concepts['value'], name='manual_concept')
    else:
        concepts = None

    question_solutions = {}
    for question in questions.itertuples():
        question_solutions[question[0]] = question[4] if question[0].replace('_', question[4]) == question[3] else question[5]
        questions.loc[question[0], 'correct'] = question_solutions[question[0]]

    questions.loc[questions['variant1'] == questions['correct'], 'correct_variant'] = 0
    questions.loc[questions['variant2'] == questions['correct'], 'correct_variant'] = 1
    if concepts:
        questions = questions.join(concepts)
    questions = questions.set_index(questions['id'])


    questions = questions[questions['concept'] == concept_id]
    print(questions)

    items = questions.loc[:, ['solved', 'manual_concept']].rename(columns={'solved': 'name', 'manual_concept': 'concept'})
    # items = questions.loc[:, ['solved', 'correct']].rename(columns={'solved': 'name', 'correct': 'concept'})
    # for f, t in (('í', 'i'), ('ý', 'y')):
    #     items.loc[items['concept'] == f, 'concept'] = t

    answers = answers.rename(columns={'id': 'item', 'user': 'student'})
    answers = answers[answers['item'].isin(items.index)]
    answers['response_time'] = 0
    answers = answers.loc[:, ['student', 'item', 'response_time', 'correct']]
    answers = answers.groupby(['student', 'item']).first().reset_index()

    print(items)
    answers.to_pickle('cestina-{}-answers.pd'.format(concept_name))
    items.to_pickle('cestina-{}-items.pd'.format(concept_name))


def math_garden_all():
    concepts = ['addition', 'subtraction', 'multiplication']
    # concepts = ['addition', 'subtraction']

    answers = pd.concat([pd.read_pickle('math_garden-{}-answers.pd'.format(c)) for c in concepts])
    answers = answers.set_index(np.arange(len(answers)))
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


# math_garden('addition')
# math_garden('multiplication')
# math_garden('subtraction')
# matmat()
# matmat_all()
# math_garden_all()
# cestina(7, 'Z')
# cestina(1, 'B')
# cestina(2, 'L')
# cestina(9, 'konc-prid')
# cestina(8, 'zs')
cestina(16, 'nn')

# simulated(n_students=100, n_concepts=2, n_items=100)
