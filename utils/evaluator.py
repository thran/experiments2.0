import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.runner import Runner


class Evaluator:
    def __init__(self, data, model):
        self._model = model
        self._data = data

        self._runner = Runner(data, model)
        self._hash = None

    def clean(self):
        self._runner.clean()

    def _evaluate(self, report, brier_bins=20):
        print("Evaluating", self._hash, self._data, self._model)

        n = 0           # log count
        sse = 0         # sum of square error
        llsum = 0       # log-likely-hood sum
        brier_counts = np.zeros(brier_bins)          # count of answers in bins
        brier_correct = np.zeros(brier_bins)        # sum of correct answers in bins
        brier_prediction = np.zeros(brier_bins)     # sum of predictions in bins

        self._data.join_predictions(pd.read_pickle(self._runner.get_log_filename()))

        for log in self._data.iter_test():
            n += 1
            sse += (log["prediction"] - log["correct"]) ** 2
            llsum += math.log(max(0.0001, log["prediction"] if log["correct"] else (1 - log["prediction"])))

            # brier
            bin = min(int(log["prediction"] * brier_bins), brier_bins - 1)
            brier_counts[bin] += 1
            brier_correct[bin] += log["correct"]
            brier_prediction[bin] += log["prediction"]

        answer_mean = sum(brier_correct) / n

        report["extra"] = {"anser_mean": answer_mean}
        report["rmse"] = math.sqrt(sse / n)
        report["log-likely-hood"] = llsum
        report["AUC"] = metrics.roc_auc_score(self._data.get_dataframe_test()["correct"], self._data.get_dataframe_test()   ["prediction"])

        # brier
        brier_prediction_means = brier_prediction / brier_counts
        brier_prediction_means[np.isnan(brier_prediction_means)] = \
            ((np.arange(brier_bins) + 0.5) / brier_bins)[np.isnan(brier_prediction_means)]
        brier_correct_means = brier_correct / brier_counts
        brier_correct_means[np.isnan(brier_correct_means)] = 0
        brier = {
            "reliability":  sum(brier_counts * (brier_correct_means - brier_prediction_means) ** 2) / n,
            "resolution":  sum(brier_counts * (brier_correct_means - answer_mean) ** 2) / n,
            "uncertainty": answer_mean * (1 - answer_mean),

        }
        report["brier"] = brier

        report["extra"]["brier"] = {
            "bin_count": brier_bins,
            "bin_counts": list(brier_counts),
            "bin_prediction_means": list(brier_prediction_means),
            "bin_correct_means": list(brier_correct_means),
        }
        report["evaluated"] = True

        self.save_report(report)
        return report

    def get_report(self, force_evaluate=False, force_run=False, **kwargs):
        self._hash = self._runner.run(force=force_run)
        report = json.load(open(self._runner.get_report_filename()))
        if force_evaluate or force_run or "evaluated" not in report:

            self._evaluate(report, **kwargs)
        return report

    def save_report(self, report):
        json.dump(report, open(self._runner.get_report_filename(), "w"), indent=4)

    def __str__(self):
        return json.dumps(self.get_report(), sort_keys=True, indent=4)

    def brier_graphs(self, show=True):
        report = self.get_report()

        plt.figure()
        plt.plot(report["zextra"]["brier"]["bin_prediction_means"], report["zextra"]["brier"]["bin_correct_means"])
        plt.plot((0, 1), (0, 1))

        bin_count = report["zextra"]["brier"]["bin_count"]
        counts = np.array(report["zextra"]["brier"]["bin_counts"])
        bins = (np.arange(bin_count) + 0.5) / bin_count
        plt.bar(bins, counts / max(counts), width=(0.5 / bin_count), alpha=0.5)
        plt.title(self._model)
        if show:
            plt.show()
