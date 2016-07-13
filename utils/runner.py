import datetime
import json
import os
import pandas as pd
from hashlib import sha1


def get_hash(model, data):
    return sha1(str(model).encode() + str(data).encode()).hexdigest()[:20]


class Runner():
    def __init__(self, data, model):
        self._data = data
        self._model = model
        self._log = {}
        self._hash = get_hash(self._model, self._data)

    def _pandas_logger(self, answer, prediction, time_prediction=None):
        self._log[answer["id"]] = [prediction, time_prediction]

    def clean(self):
        os.remove(self.get_log_filename())
        os.remove(self.get_report_filename())

    def run(self, force=False, only_train=False, skip_pre_process=False):
        if not force and (os.path.exists(self.get_log_filename()) and os.path.exists(self.get_report_filename())):
            print("Report and log in cache - {} ".format(self._hash))
            return self._hash

        if not skip_pre_process:
            start = datetime.datetime.now()
            print("Pre-processing data...")
            self._model.pre_process_data(self._data)
            pre_processing_time = datetime.datetime.now() - start
            print("  total runtime:", pre_processing_time)
        else:
            pre_processing_time = None

        start = datetime.datetime.now()
        print("Processing data...")
        self._model.process_data(self._data, self._pandas_logger, only_train=only_train)
        processing_time = datetime.datetime.now() - start
        print("  total runtime:", processing_time)

        report = {
            "model": str(self._model),
            "data": str(self._data),
            "processing time": str(processing_time),
            "pre-processing time": str(pre_processing_time),
            "dataset size": len(self._data.get_dataframe_all()),
        }
        json.dump(report, open(self.get_report_filename(), "w"), indent=4)
        pd.DataFrame.from_dict(self._log, orient='index').to_pickle(self.get_log_filename())

        print("Report and log written to cache - {} ".format(self._hash))
        return self._hash

    def get_log_filename(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "{}.log.pd".format(self._hash))

    def get_report_filename(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "{}.report.json".format(self._hash))

