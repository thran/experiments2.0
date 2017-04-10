import numpy as np
import pandas as pd

from models.eloConcepts import EloConcepts
from utils.data import items_in_concept
from utils.runner import Runner
from utils.utils import Cache


def add_predictions(data, model):
    r = Runner(data, model)
    r.run()
    data.join_predictions(pd.read_pickle(r.get_log_filename()))


@Cache(hash=True, type='pandas')
def get_skills(data, model, cache=None, name='skill'):
    skills = pd.Series(index=data.get_dataframe_all()['id'], name=name)

    def callback(answer):
        if type(model) == EloConcepts:
            skills[answer['id']] = model.get_skill(answer['student'], concept=model._get_concept(answer['item']))
        else:
            skills[answer['id']] = model.get_skill(answer['student'])

    model.after_update_callback = callback
    r = Runner(data, model)
    r.run(force=True)
    return skills


def add_skills(data, model, name='skill', data_to_join=None):
    if data_to_join is None:
        data_to_join = data
    skills = get_skills(data, model, cache='{}-{}-'.format(data, model, name), name=name)
    df = data_to_join.get_dataframe_test()
    data_to_join._data_test = df.join(skills, on="id")


@Cache(hash=True, type='pandas')
def get_difficulties(data, model, cache=None, name='difficulty'):
    r = Runner(data, model)
    r.run(force=True)
    items = data.get_dataframe_all()['item'].unique()
    return pd.Series(data=model.get_difficulties(items), index=items, name=name)


def add_difficulties(data, model, name='difficulty', data_to_join=None):
    if data_to_join is None:
        data_to_join = data
    skills = get_difficulties(data, model, cache='{}-{}-'.format(data, model, name), name=name)
    df = data_to_join.get_dataframe_test()
    data_to_join._data_test = df.join(skills, on="item")


@Cache(hash=True, type='json')
def get_mastery_curves(data, concept, mastery_metric, min_answers=11, cache=None):
    answers = data.get_dataframe_test()
    answers = answers[answers['item'].isin(items_in_concept(data, concept))]

    sessions = answers.groupby('session').apply(len)
    sessions = sessions[sessions >= min_answers]

    curves = []
    for session in sorted(sessions.index):
        session_answers = answers[answers['session'] == session]
        curves.append(mastery_metric(session_answers))
    return curves


@Cache(hash=True, type='pandas')
def thresholds(curves, count=100, cache=None):
    data = pd.DataFrame(columns=np.arange(1, count + 1) / count)
    for curve in curves:
        d = []
        for i, c in enumerate(curve):
            d += [i] * max(0, int(c * 100) - len(d))
        d += [None] * (count - len(d))
        data.loc[len(data)] = d
    return data


def get_order(curves):
    # return np.argsort(np.argsort(curves))
    return np.argsort(np.argsort([np.mean(c[-5:]) for c in curves]))

