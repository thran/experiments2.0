import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from utils.evaluator import Evaluator
import math


def compare_models(data, models, dont=False, answer_filters=None, metric="rmse",evaluate=False, diff_to=None, force_evaluate=False, force_run=False):
    if dont:
        return
    df = pd.DataFrame(columns=["model", "data", metric])
    for model in models:
        report = Evaluator(data, model).get_report(
            force_evaluate=force_evaluate,
            force_run=force_run,
            answer_filters=answer_filters,
        )
        for filter_name in ["all"] + list(answer_filters.keys()):
            if filter_name == "all":
                r = report
            else:
                r = report[filter_name]
            print(model, filter_name)
            print("RMSE: {:.5}".format(r["rmse"]))
            if diff_to is not None:
                print("RMSE diff: {:.5f}".format(diff_to - r["rmse"]))
            print("LL: {:.6}".format(r["log-likely-hood"]))
            print("AUC: {:.4}".format(r["AUC"]))
            print("Brier resolution: {:.4}".format(r["brier"]["resolution"]))
            print("Brier reliability: {:.3}".format(r["brier"]["reliability"]))
            print("Brier uncertainty: {:.3}".format(r["brier"]["uncertainty"]))
            print("=" * 50)

            df.loc[len(df)] = (str(model), filter_name, r[metric])

    sns.barplot(x="data", y=metric, hue="model", data=df,
                hue_order=sorted(df["model"].unique(), key=lambda i: df[df["model"]==i][metric].mean()),
                order=["all"] + sorted(list(answer_filters.keys())))
    plt.ylim((math.floor(100 * df[metric].min()) / 100, math.ceil(100 * df[metric].max()) / 100))