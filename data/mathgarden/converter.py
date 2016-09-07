from rpy2.robjects import r, pandas2ri


def convert(name):
    r.load('{}.RData'.format(name))
    pandas2ri.ri2py(r['res_max2']).to_pickle('{}.pd'.format(name))
    print(pandas2ri.ri2py(r['res_max2']).head())

convert('addition')
# convert('multiplication')
# convert('subtraction')
