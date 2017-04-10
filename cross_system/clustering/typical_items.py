import pandas as pd
import numpy as np

from cross_system.clustering.similarity import similarity_pearson

data_set = 'cestina-B'
# data_set = 'cestina-zs'
# data_set = 'cestina-nn'
# similarity = lambda x: similarity_pearson(similarity_pearson(x))
similarity = lambda x: similarity_pearson(similarity_pearson(x))

answers = pd.read_pickle('data/{}-answers.pd'.format(data_set))
items = pd.read_pickle('data/{}-items.pd'.format(data_set))


X = similarity(answers)
for item in X.index:
    X.loc[item, item] = 0

# threshold = len(X.columns) // 5
threshold = 10

df = pd.DataFrame(index=X.index)
df['mean'] = X.mean()
df['name'] = X.index
df['name'] = df['name'].apply(lambda x: items.loc[x]['name'])
df['success_rate'] = answers.groupby('item')['correct'].mean()
df['answer_count'] = answers.groupby('item').apply(len)

for item, values in X.iterrows():
    df.loc[item, 'top_{}'.format(threshold)] = np.mean(sorted(values.values)[-threshold:])

# df = df.sort('mean', ascending=False)
df = df.sort('top_{}'.format(threshold), ascending=False)

pd.set_option('display.max_rows', len(df))
pd.set_option('expand_frame_repr', False)
print(df)
print(df.loc[:, ['mean', 'top_{}'.format(threshold)]].corr(method='spearman'))
