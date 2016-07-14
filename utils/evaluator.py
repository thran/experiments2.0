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

    def evaluate(self, force_evaluate=False, answer_filters=None, **kwargs):
        answer_filters = answer_filters if answer_filters is not None else {}
        report = self._load_report()
        self._data.join_predictions(pd.read_pickle(self._runner.get_log_filename()))
        if force_evaluate or "evaluated" not in report:
            print("Evaluating", self._hash, self._data, self._model)
            report.update(self._basic_metrics(self._data.iter_test(), **kwargs))

            report['time'] = self._basic_metrics(
                self._data.iter_test(),
                prediction_column="time_prediction_log",
                observation_column="response_time_log",
                brier_min=self._data.get_dataframe_test()['time_prediction_log'].min(),
                brier_max=self._data.get_dataframe_test()['time_prediction_log'].max(),
                **kwargs)

            report['time-raw'] = self._basic_metrics(
                self._data.iter_test(),
                prediction_column="time_prediction",
                observation_column="response_time",
                brier_min=self._data.get_dataframe_test()['time_prediction'].min(),
                brier_max=self._data.get_dataframe_test()['time_prediction'].max(),
                **kwargs)

        if answer_filters is not None:
            for filter_name, filter_function in answer_filters.items():
                if force_evaluate or filter_name not in report:
                    print("Evaluating", filter_name, self._hash, self._data, self._model)
                    data = filter_function(self._data.get_dataframe_test())
                    report[filter_name] = self._basic_metrics(self._data.iter(data=data), **kwargs)

        self._save_report(report)
        return report

    def _basic_metrics(self, data, brier_bins=20, prediction_column="prediction", observation_column="correct", brier_min=0, brier_max=1):
        report = {}

        n = 0           # log count
        sse = 0         # sum of square error
        llsum = 0       # log-likely-hood sum
        brier_counts = np.zeros(brier_bins)          # count of answers in bins
        brier_correct = np.zeros(brier_bins)        # sum of correct answers in bins
        brier_prediction = np.zeros(brier_bins)     # sum of predictions in bins

        for log in data:
            n += 1
            sse += (log[prediction_column] - log[observation_column]) ** 2
            llsum += math.log(max(0.0001, log[prediction_column] if log[observation_column] else (1 - log[prediction_column])))

            # brier
            bin = min(int((log[prediction_column] - brier_min) / (brier_max - brier_min) * brier_bins), brier_bins - 1)
            brier_counts[bin] += 1
            brier_correct[bin] += log[observation_column]
            brier_prediction[bin] += log[prediction_column]

        answer_mean = sum(brier_correct) / n

        report["extra"] = {"answer_mean": answer_mean}
        report["rmse"] = math.sqrt(sse / n)
        report["log-likely-hood"] = llsum
        if observation_column == "correct":
            try:
                report["AUC"] = metrics.roc_auc_score(self._data.get_dataframe_test()[observation_column],
                                                      self._data.get_dataframe_test()[prediction_column])
            except ValueError:
                print("AUC - converting responses to 0, 1")
                report["AUC"] = metrics.roc_auc_score(self._data.get_dataframe_test()[observation_column] > 0,
                                                      self._data.get_dataframe_test()[prediction_column])

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
            "max": brier_max,
            "min": brier_min,
            "bin_count": brier_bins,
            "bin_counts": list(brier_counts),
            "bin_prediction_means": list(brier_prediction_means),
            "bin_correct_means": list(brier_correct_means),
        }
        report["evaluated"] = True

        return report

    def get_report(self, force_evaluate=False, force_run=False, **kwargs):
        self._hash = self._runner.run(force=force_run)
        return self.evaluate(force_evaluate=force_evaluate or force_run, **kwargs)

    def roc_curve(self):
        self.get_report()
        self._data.join_predictions(pd.read_pickle(self._runner.get_log_filename()))
        fpr, tpr, thresholds = metrics.roc_curve(self._data.get_dataframe_test()["correct"] > 0, self._data.get_dataframe_test()["prediction"])
        print(fpr, tpr, thresholds)
        plt.plot(fpr, tpr, label=str(self._data))

    def _save_report(self, report):
        json.dump(report, open(self._runner.get_report_filename(), "w"), indent=4)

    def _load_report(self):
        return json.load(open(self._runner.get_report_filename()))

    def __str__(self):
        return json.dumps(self.get_report(), sort_keys=True, indent=4)

    def brier_graphs(self, time=False, time_raw=False):
        report = self.get_report()
        if time and not time_raw:
            report = report['time']
        if time_raw:
            report = report['time-raw']

        plt.figure()
        plt.plot(report["extra"]["brier"]["bin_prediction_means"], report["extra"]["brier"]["bin_correct_means"])
        l = report["extra"]["brier"]['min'], report["extra"]["brier"]['max']
        plt.plot(l, l)

        bin_count = report["extra"]["brier"]["bin_count"]
        counts = np.array(report["extra"]["brier"]["bin_counts"])
        bins = (np.arange(bin_count) + 0.5) * (l[1] - l[0]) / bin_count + l[0]
        plt.bar(bins, counts / max(counts) * l[1], width=(0.5 / bin_count * (l[1] - l[0])), alpha=0.5)
        plt.title(self._model)
        plt.xlabel('prediction')
        plt.ylabel('observation mean')
