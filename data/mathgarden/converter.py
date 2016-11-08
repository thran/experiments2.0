import pandas as pd
from rpy2.robjects import r, pandas2ri


def convert_from_R(name):
    r.load('{}.RData'.format(name))
    df = pandas2ri.ri2py(r['res_max2'])
    df = convert_to_data_format(df)
    df.to_pickle('{}.pd'.format(name))
    return df

def items():
    r.load('{}.rdata'.format('items123'))
    d = pandas2ri.ri2py(r['items123'])
    s = pd.Series(index=map(int, d[:len(d) // 2]), data=d[len(d) // 2:])
    s.to_pickle('items.pd')
    return s

def convert_to_data_format(df):
    df = df.rename(columns={
        'user_id': 'student',
        'item_id': 'item',
        'correct_answered': 'correct',
        'response_in_milliseconds': 'response_time',
    })
    df['response_time'] /= 1000
    df['student'] = df['student'].astype(int)
    df['item'] = df['item'].astype(int)
    df['id'] = df.index
    del df['score']
    del df['days']
    del df['created_UNIX']
    return df

# items()

convert_from_R('addition')
print(convert_from_R('multiplication'))
convert_from_R('subtraction')


