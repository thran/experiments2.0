import pandas as pd


problems = ['binary-crosswords', 'nurikabe','region-puzzle', 'robotanic', 'rush-hour', 'slitherlink', 'sokoban', 'title-maze']


def process(filename):

    df = pd.read_csv(filename)
    df["Login"] = df["Login"].str.replace("U", '').astype(int)
    df.set_index(df["Login"], inplace=True)

    counts = (~df.isnull()).sum(axis=1)
    df = df[counts>2]
    df.to_pickle(filename.replace('csv', 'pd'))
    print('{}: {} items, {} learners, {} answers'.format(filename, len(df.columns), len(df.index), (~df.isnull()).sum().sum()))


# for problem in problems:
#     process(problem + '.csv')