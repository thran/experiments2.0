import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from models.eloPriorCurrent import EloPriorCurrentModel
from utils.data import Data, TimeLimitResponseModificator, ExpDrop, LinearDrop, transform_response_by_time, \
    filter_students_with_many_answers, response_as_binary, transform_response_by_time_linear, \
    items_in_concept
from utils.runner import Runner


def difficulty_stability(datas, models, labels, points, runs=1, eval_data=None, data_ratio=1.):
    filename = "../cache/tmp4.data.pd"
    df = pd.DataFrame(columns=["answers", "correlation", "models"])
    student_count = len(datas[0](None).get_dataframe_all())
    for i in range(points):
        ratio = (i + 1) / points
        print("\n\nEvaluation for {}% of data\n\n".format(ratio * 100 * data_ratio))

        for data, model, label in zip(datas, models, labels):
            for run in range(runs):
                d = data(None)
                d.set_seed(run)
                d.set_train_size(ratio * data_ratio)
                d.filter_data(100, 0)
                d.get_dataframe_train().to_pickle(filename)
                m1 = model(None)
                m2 = model(None)
                d1 = Data(filename, train_size=0.5, train_seed=run + 42)
                d2 = Data(filename, train_size=0.5, train_seed=-run - 42)

                Runner(d1, m1).run(force=True, only_train=True)
                Runner(d2, m2).run(force=True, only_train=True)

                items = d.get_items()
                if eval_data is None:
                    v1 = pd.Series(m1.get_difficulties(items), items)
                    v2 = pd.Series(m2.get_difficulties(items), items)
                else:
                    r1 = Runner(eval_data(None), m1)
                    r2 = Runner(eval_data(None), m2)
                    r1.run(force=True, skip_pre_process=True)
                    r2.run(force=True, skip_pre_process=True)
                    v1 = pd.Series(r1._log)
                    v2 = pd.Series(r2._log)
                df.loc[len(df)] = (ratio * 100 * data_ratio, v1.corr(v2), label)

    print(df)
    sns.factorplot(x="answers", y="correlation", hue="models", data=df) # markers=["o", "^", "v", "s", "D"]
    return df


filename, ratio, items = "../data/matmat/2016-11-28/answers.pd", 1, pd.read_csv('../data/matmat/2016-11-28/items.csv', index_col='id')['question']
# filename, ratio, items = "../data/mathgarden/multiplication.pd", 0.2, pd.read_pickle("../data/mathgarden/items.pd")
# filename, ratio, items = "../data/mathgarden/addition.pd", 0.1, pd.read_pickle("../data/mathgarden/items.pd")
# filename, ratio, items = "../data/mathgarden/subtraction.pd", 0.1, pd.read_pickle("../data/mathgarden/items.pd")
data = lambda l: Data(filename, train_size=0.7)
basic_model = lambda label: EloPriorCurrentModel(KC=2, KI=0.5)

if 0:
    difficulty_stability(
        [
            data,
            lambda l: Data(filename, response_modification=TimeLimitResponseModificator([(7, 0.5)])),
            lambda l: Data(filename, response_modification=ExpDrop(5, 0.9)),
            lambda l: Data(filename, response_modification=LinearDrop(14)),
        ],
        [basic_model, basic_model, basic_model, basic_model],
        ["Basic model + noTime", "Basic model + thresholdTme", "Basic model + expTime", "Basic model + linearTime"],
        10, runs=5, data_ratio=ratio,
        # eval_data=data_test
    )

if 1:
    ratio = 1
    model1 = basic_model(None)
    model2 = basic_model(None)
    data1 = Data(filename, train_size=ratio)
    median = data1.get_dataframe_all()['response_time'].median()
    print('time median', median)
    data2 = Data(filename, response_modification=LinearDrop(median * 2), train_size=ratio)
    # data2 = Data(filename, response_modification=TimeLimitResponseModificator([(median, 0.5)]), train_size=ratio)
    # data2 = Data(filename, response_modification=ExpDrop(median / 2, 0.9), train_size=ratio)

    Runner(data1, model1).run(force=True, only_train=True)
    Runner(data2, model2).run(force=True, only_train=True)

    items_ids = data1.get_items()
    items_ids = list(items_in_concept(data(None), 'division'))

    v1 = model1.get_difficulties(items_ids)
    v2 = model2.get_difficulties(items_ids)
    for item, x, y in zip(items_ids, v1, v2):
        plt.plot(x, y, ".")
        plt.text(x, y, items.loc[item])
    plt.xlabel(str(data1))
    plt.ylabel(str(data2))

plt.show()
