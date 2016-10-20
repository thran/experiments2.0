import pandas as pd
from rpy2.robjects import r, pandas2ri


def convert(name):
    r.load('{}.RData'.format(name))
    df = pandas2ri.ri2py(r['res_max2'])
    df.to_pickle('{}.pd'.format(name))

def items():
    r.load('{}.rdata'.format('items123'))
    d = pandas2ri.ri2py(r['items123'])
    s = pd.Series(index=map(int, d[:len(d) // 2]), data=d[len(d) // 2:])
    s.to_pickle('items.pd')
    return s


items()

# convert('addition')
# convert('multiplication')
# convert('subtraction')


