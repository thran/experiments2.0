import pandas as pd


data = pd.read_csv('../data/matmat/2016-06-27/answers.csv')
items = pd.read_csv('../data/matmat/2016-06-27/items.csv')
data = data.join(items, on='item', rsuffix='_item')
data = data[~data['skill_lvl_1'].isnull()]
data['skill_lvl_1'] = data['skill_lvl_1'].astype('int')

data = data.loc[:, ['student', 'skill_lvl_1', 'correct', 'timestamp']]
data = data.rename(columns={'student': 'user', 'skill_lvl_1': 'context', 'timestamp': 'time'})

# data.to_csv('../data/matmat/matmat.csv', index=False)
print(data['context'].unique())
