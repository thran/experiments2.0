import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

df = pd.read_pickle('../data/matmat/2016-06-27/answers.pd')
# df = df.loc[:10000]


def trim_times(df,columns="response_time", limit=30):
    df.loc[df[columns] < 0.5, columns] = 0.5
    df.loc[df[columns] > limit, columns] = limit


def pandas_shift(df):
    def time_shift(df):
        df['response_time_last'] = df['response_time'].shift()
        return df

    df = df.groupby(['student', 'item']).apply(time_shift)
    df.to_pickle("matmat.shifted.pd")


def pandas_shift_student(df):
    def time_shift(df):
        df['response_time_last'] = df['response_time'].shift()
        return df

    df = df.groupby(['student']).apply(time_shift)
    df.to_pickle("matmat.shifted_student.pd")
    print(df)


# pandas_shift_student(df)
df = pd.read_pickle("matmat.shifted_student.pd")
df = df[~df['response_time_last'].isnull()]
trim_times(df)
trim_times(df, 'response_time_last')

df['response_time_dif'] = - df['response_time_last'] + df['response_time']
# plt.plot(df['response_time'], df['response_time_dif'], '.')
# sns.jointplot("response_time", "response_time_last", data=df, kind="hex")
# sns.jointplot("response_time_last", "response_time_dif", data=df, kind="kde")
print(df['response_time_dif'].mean())
print(df['response_time_dif'].median())
print(np.abs(df['response_time_dif']).mean())
print(np.abs(df['response_time_dif']).median())

plt.show()
