import copy

import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from utils.evaluator import Evaluator
import math


def compare_models(data, models, names=None, dont=False, answer_filters=None, metric="rmse",evaluate=False,
                   diff_to=None, force_evaluate=False, force_run=False, runs=1, hue_order=True, with_all=True, **kwargs):
    if dont:
        return
    if answer_filters is None:
        answer_filters = {}
    df = pd.DataFrame(columns=["model", "data", metric, 'time-'+metric])
    datas = [data] * len(models) if type(data) is not list else data
    for i, (dat, model) in enumerate(zip(datas, models)):
        for run in range(runs):
            if runs > 1:
                d = copy.deepcopy(dat)
                m = copy.deepcopy(model)
                d.set_seed(run)
            else:
                d = dat
                m = model
            report = Evaluator(d, m).get_report(
                force_evaluate=force_evaluate,
                force_run=force_run,
                answer_filters=answer_filters,
            )
            for filter_name in (["all"] if with_all else []) + list(answer_filters.keys()):
                if filter_name == "all":
                    r = report
                else:
                    r = report[filter_name]
                print(model, filter_name)
                print("RMSE: {:.5}".format(r["rmse"]))
                print("RMSE-time: {:.5}".format(r['time']["rmse"]))
                if diff_to is not None:
                    print("RMSE diff: {:.5f}".format(diff_to - r["rmse"]))
                print("LL: {:.6}".format(r["log-likely-hood"]))
                if "AUC" in r:
                    print("AUC: {:.4}".format(r["AUC"]))
                print("Brier resolution: {:.4}".format(r["brier"]["resolution"]))
                print("Brier reliability: {:.3}".format(r["brier"]["reliability"]))
                print("Brier uncertainty: {:.3}".format(r["brier"]["uncertainty"]))
                print("=" * 50)

                model_name = (str(model) if type(data) is not list else str(model) + str(d))[:50]
                if names is not None:
                    model_name = names[i]
                df.loc[len(df)] = (model_name, filter_name, r[metric], r['time'][metric])

    print(df)

    for m in [metric, 'time-'+metric]:
        plt.figure()
        if type(data) is not list:
            plt.title(data)
        hue_order = sorted(df["model"].unique(), key=lambda i: df[df["model"]==i][m].mean()) if hue_order else None

        sns.barplot(x="data", y=m, hue="model", data=df,
                    hue_order=hue_order,
                    order=(["all"] if with_all else []) + sorted(list(answer_filters.keys())), **kwargs)
        plt.ylim((math.floor(100 * df[m].min()) / 100, math.ceil(100 * df[m].max()) / 100))