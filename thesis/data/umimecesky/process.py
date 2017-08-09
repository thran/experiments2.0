import pandas as pd

def answers():
    files = [
        'doplnovacka_log_2016-09-01_2016-11-01.csv',
        'doplnovacka_log_2016-10-01_2017-01-01.csv',
        'doplnovacka_log_2017-01-01_2017-03-01.csv',
        'doplnovacka_log_2017-03-01_2017-05-01.csv',
    ]


    df = pd.concat([pd.read_csv(file, sep=';') for file in files])
    df = df.sort_values('time')
    df.reset_index(inplace=True)

    print(df.columns)


    df.rename(columns={
        "time": "timestamp",
        "word": "item",
        "user": "student",
        "responseTime": "response_time",
    }, inplace=True)

    df['response_time'] = df['response_time'] / 1000
    df['response_time'][df['response_time'] == 0] = None

    del df['index']

    print(df)
    df.to_pickle('answers.pd')

answers()


def items():
    items = pd.read_csv('doplnovacka_word.csv', sep=';', index_col='id')
    map = pd.read_csv('doplnovacka_concept_word.csv', sep=';', index_col='word')
    concepts = pd.read_csv('concepts.csv', sep=';', index_col='id')

    items = items.merge(map, left_index=True, right_index=True)
    items = items.merge(concepts, left_on='concept', right_index=True)

    del items['explanation']
    del items['Unnamed: 2_x']
    del items['Unnamed: 2_y']

    print(items)

# items()