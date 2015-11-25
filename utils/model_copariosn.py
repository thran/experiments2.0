import matplotlib.pylab as plt
from utils.evaluator import Evaluator


def compare_models(data, models, dont=False, resolution=True, auc=False, evaluate=False, diff_to=None, force_evaluate=False, force_run=False):
    if dont:
        return
    plt.xlabel("RMSE")
    if auc:
        plt.ylabel("AUC")
    elif resolution:
        plt.ylabel("Resolution")
    else:
        plt.ylabel("Brier score")
    for model in models:
        report = Evaluator(data, model).get_report(force_evaluate=force_evaluate, force_run=force_evaluate)
        print(model)
        print("RMSE: {:.5}".format(report["rmse"]))
        if diff_to is not None:
            print("RMSE diff: {:.5f}".format(diff_to - report["rmse"]))
        print("LL: {:.6}".format(report["log-likely-hood"]))
        print("AUC: {:.4}".format(report["AUC"]))
        print("Brier resolution: {:.4}".format(report["brier"]["resolution"]))
        print("Brier reliability: {:.3}".format(report["brier"]["reliability"]))
        print("Brier uncertainty: {:.3}".format(report["brier"]["uncertainty"]))
        print("=" * 50)

        x = report["rmse"]
        if auc:
            y = report["AUC"]
        elif resolution:
            y = report["brier"]["resolution"]
        else:
            y = report["brier"]["reliability"] - report["brier"]["resolution"] + report["brier"]["uncertainty"]
        plt.plot(x, y, "bo")
        plt.text(x, y, model, rotation=0, )