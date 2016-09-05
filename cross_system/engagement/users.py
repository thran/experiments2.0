import os

import numpy as np
import seaborn as sns
import pandas as pd
from pandas.tseries.offsets import Minute
import matplotlib.pylab as plt


def session_count(df):
    next_timestamp = df.groupby("user")["time"].shift(-1)
    last_in_session = next_timestamp.isnull() | (next_timestamp - df["time"] > pd.Timedelta(Minute(30)))
    return sum(last_in_session)


def user_stats(data_set='matmat', force=False):
    file_path = '../../cache/engagement/{}.pd'.format(data_set)

    if os.path.exists(file_path) and not force:
        print('Loading cached user stats')
        return pd.read_pickle(file_path)

    df = pd.read_csv('../../data/engagement/{}.csv'.format(data_set))
    df["time"] = pd.to_datetime(df["time"])
    users = pd.DataFrame(index=df['user'].unique())

    users['answers'] = df.groupby('user').apply(len)
    users['sessions'] = df.groupby('user').apply(session_count)
    users['success_rate'] = df.groupby('user')['correct'].mean()

    users.to_pickle(file_path)
    return(users)

# df = user_stats('matmat', force=False)
# df = user_stats('umimecesky-doplnovacka', force=False)
df = user_stats('slepemapy', force=False)
df = df[df['answers'] >= 10]
df['answers'] = np.log10(df['answers'])
df['sessions'] = np.log10(df['sessions'])
# df.loc[df['sessions'] > 10]


g = sns.PairGrid(df, diag_sharey=False)
g.map_diag(plt.hist)
g.map_upper(plt.scatter, marker=".")
g.map_lower(sns.kdeplot, shade=False, cmap="Blues_d")

plt.show()