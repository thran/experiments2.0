import json
from collections import defaultdict

import pandas as pd
from sklearn import linear_model

from cross_system.clustering.clusterings import kmeans
from cross_system.clustering.similarity import similarity_pearson
from utils.data import Data

df_types = pd.read_csv('place_type.csv', sep=';')
df_types.set_index(df_types['id'], inplace=True)
# print(df_types)

def get_one_concept(concept='Asia', type=1):

    items = pd.read_csv('items.csv', sep=';')
    id = items[items['name'] == concept]['id'].values[0]
    items = items[items['type'] == type]

    print(id)
    print('')

    ids = {}
    for _, item in items.iterrows():
        if id in json.loads(item['maps']):
            print(item['id'], item['name'])
            ids[item['id']] = item['name']

    json.dump(ids, open('{}.json'.format(concept), 'w'))
    return ids

def concepts_types():
    items = pd.read_csv('items.csv', sep=';')
    types = items['type'].unique()
    print(types, len(types))

    concepts = {}
    for type in types:
        concepts[df_types.get_value(type, 'name')] = list(map(int, items[items['type'] == type]['id'].values))

    print(concepts)
    json.dump(concepts, open('concepts_types.json', 'w'))
    return concepts

def concepts_locations():
    items = pd.read_csv('items.csv', sep=';', index_col='id')

    concepts = defaultdict(lambda: [])
    for id, item in items.iterrows():
        for concept in json.loads(item['maps'])[:1]:
            concept_name = items.get_value(concept, 'name')
            concepts[concept_name].append(int(id))

    print(list(concepts.keys()), len(list(concepts.keys())))

    print(concepts)
    json.dump(concepts, open('concepts_locations.json', 'w'))
    return concepts


def concepts_combinations():
    types = {}
    for type, items in json.load(open('concepts_types.json')).items():
        for item in items:
            types[item] = type

    concepts = defaultdict(lambda: [])
    for location, items in json.load(open('concepts_locations.json')).items():
        for item in items:
            concepts['{} -- {}'.format(location, types[item])].append(item)

    print(list(concepts.keys()), len(list(concepts.keys())))

    print(concepts)
    json.dump(concepts, open('concepts_combinations.json', 'w'))
    return concepts


def concepts_automatic(concepts_count):
    X = pd.read_pickle('../../cache/get_similarity-319e002b0b6b1575dbd5.pd')        # hacl

    concepts = defaultdict(lambda: [])
    for item, c in zip(X.index, kmeans(X, concepts_count)):
        concepts[str(c)].append(int(item))
    print(concepts)
    json.dump(concepts, open('concepts_{}.json'.format(concepts_count), 'w'))
    return concepts

def concepts_corrected(concepts_name):
    concepts = json.load(open('concepts_{}.json'.format(concepts_name)))
    X = pd.read_pickle('../../cache/get_similarity-319e002b0b6b1575dbd5.pd')        # hacl
    labels = []
    keys = list(concepts.keys())
    for item in X.index:
        for k, v in concepts.items():
            if item in v:
                labels.append(keys.index(k))
                break
        else:
            print(item)
            labels.append(len(keys))


    clf = linear_model.LogisticRegression(C=2)
    print(len(labels), X.shape)
    clf.fit(X, labels)
    corrected_labels = clf.predict(X)

    new_concepts = defaultdict(lambda: [])
    for item, l in zip(X.index, corrected_labels):
        new_concepts[str(l)].append(int(item))
    json.dump(new_concepts, open('concepts_{}_corrected.json'.format(concepts_name), 'w'))
    return new_concepts

# concepts_types()
# concepts_locations()
# concepts_combinations()
# concepts_automatic(5)
# concepts_automatic(20)
# concepts_automatic(50)

# concepts_corrected('types')
# concepts_corrected('locations')
# concepts_corrected('combinations')
