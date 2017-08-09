import json
from collections import defaultdict

import pandas as pd

df = pd.read_pickle('answers.pd')
df.drop_duplicates('item', inplace=True)
# print(df)
# exit()

concepts = defaultdict(lambda: [])

for _, item in df.iterrows():
    concepts['unknown' if item['systems_asked'] == 'unknown' else json.loads(item['systems_asked'])[0]].append(item['item'])
json.dump(concepts, open('{}.json'.format('concepts_systems'), 'w'))

concepts = defaultdict(lambda: [])

for _, item in df.iterrows():
    concepts['unknown' if item['locations_asked'] == 'uknown' else json.loads(item['locations_asked'])[0]].append(item['item'])
json.dump(concepts, open('{}.json'.format('concepts_locations'), 'w'))


concepts = defaultdict(lambda: [])

for _, item in df.iterrows():
    location = 'unknown' if item['locations_asked'] == 'uknown' else json.loads(item['locations_asked'])[0]
    system = 'unknown' if item['systems_asked'] == 'unknown' else json.loads(item['systems_asked'])[0]
    concepts['{} -- {}'.format(location, system)].append(item['item'])
json.dump(concepts, open('{}.json'.format('concepts_combinations'), 'w'))