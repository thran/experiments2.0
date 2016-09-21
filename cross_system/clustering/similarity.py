import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from skll.metrics import kappa

from utils.utils import cache_pandas


def yulesQ(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    OR = (a * d) / (b * c)
    return (OR - 1) / (OR + 1)


def links(x, y):
    return (x & y).sum()


def remove_nans(df, to_zero=True):
    if to_zero:
        df[np.isnan(df)] = 0
        return df

    filter = np.isnan(df).sum() < len(df) / 2
    df = df.loc[filter, filter]
    while np.isnan(df).sum().sum() > 0:
        worst = np.isnan(df).sum().argmax()
        df = df.loc[df.index != worst, df.index != worst]
    return df


def pairwise_metric(df, metric, min_periods=1, prefect_fit=1.):
    mat = df.as_matrix().T
    K = len(df.columns)
    met = np.empty((K, K), dtype=float)
    mask = np.isfinite(mat)

    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue

            valid = mask[i] & mask[j]
            if valid.sum() < min_periods:
                c = np.nan
            elif i == j:
                c = prefect_fit
            elif not valid.all():
                c = metric(ac[valid], bc[valid])
            else:
                c = metric(ac, bc)
            met[i, j] = c
            met[j, i] = c
    return remove_nans(pd.DataFrame(met, index=df.columns, columns=df.columns))


@cache_pandas
def similarity_pearson(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return remove_nans(data.corr())


@cache_pandas
def similarity_kappa(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, kappa)


@cache_pandas
def similarity_cosine(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, cosine)


@cache_pandas
def similarity_euclidean(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, euclidean, prefect_fit=0.)


@cache_pandas
def similarity_yulesQ(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, yulesQ)


def similarity_links(data, trash_hold=0.1):
    return pairwise_metric(data > trash_hold, links)

def similarity_double_pearson(answers):
    return similarity_pearson(similarity_pearson(answers))
