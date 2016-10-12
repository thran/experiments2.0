import pandas as pd
from rpy2.robjects import r, pandas2ri


def convert(name):
    r.load('{}.RData'.format(name))
    pandas2ri.ri2py(r['res_max2']).to_pickle('{}.pd'.format(name))
    print(pandas2ri.ri2py(r['res_max2']).head())

def items():
    r.load('{}.rdata'.format('items123'))
    d = pandas2ri.ri2py(r['items123'])
    pd.Series(index=map(int, d[:len(d) // 2]), data=d[len(d) // 2:]).to_pickle('items.pd')


items()

# convert('addition')
# convert('multiplication')
# convert('subtraction')


