import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from skll.metrics import kappa
import seaborn as sns
from utils.utils import cache
import matplotlib.pylab as plt


def yulesQ(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    OR = (a * d) / (b * c)
    return (OR - 1) / (OR + 1)


def accuracy(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    return (a + d) / (a + b + c +d)


def jaccard(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    return (a) / (a + b + c)


def sokal(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    return (a + d) / (a + b + c +d)


def ochiai(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()

    return (a) / math.sqrt((a + b) * (a + c))



def kappa_own(x, y):
    a = ((x==1) & (y==1)).sum()
    b = ((x==1) & (y==0)).sum()
    c = ((x==0) & (y==1)).sum()
    d = ((x==0) & (y==0)).sum()
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (b + d) * (c + d)) / (n ** 2)

    return (po - pe) / (1 - pe)


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


@cache
def similarity_pearson(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return remove_nans(data.corr())


@cache
def similarity_kappa(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, kappa_own)


@cache
def similarity_kappa2(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, kappa)


@cache
def similarity_cosine(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, cosine)


@cache
def similarity_euclidean(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, euclidean, prefect_fit=0.)


@cache
def similarity_yulesQ(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, yulesQ)


@cache
def similarity_ochiai(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, ochiai)


@cache
def similarity_sokal(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, sokal)


@cache
def similarity_accuracy(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, accuracy)


@cache
def similarity_jaccard(data, cache=None):
    if 'student' in data.columns:
        data = data.pivot('student', 'item', 'correct')
    return pairwise_metric(data, jaccard)


def similarity_links(data, trash_hold=None):
    if trash_hold is None:
        trash_hold = data.median()
    return pairwise_metric(data > trash_hold, links)

def similarity_double_pearson(answers):
    return similarity_pearson(similarity_pearson(answers))

def plot_similarity_hist(X, ground_truth, similarity_name):
    same, different = [], []
    for concept1 in set(ground_truth):
        for concept2 in set(ground_truth):
            values = list(X.loc[ground_truth == concept1, ground_truth == concept2].values.flatten())
            if concept1 == concept2:
                same += values
            elif concept1 > concept2:
                different += values

    if similarity_name.endswith('links'):
        sns.distplot(same)
        if len(different):
            sns.distplot(different)
    elif not similarity_name.endswith('euclid'):
        plt.xlim([-1,1])
        sns.distplot(same)
        if len(different):
            sns.distplot(different)
    else:
        if len(different):
            plt.xlim([-max(different), 0])
            sns.distplot(-np.array(different))
        sns.distplot(-np.array(same))
