from collections import defaultdict

import numpy as np

from models.model import Model


class SkipHandler(Model):
    """
    Model wrapper which deals with repetitive skipped answers
    """

    def __init__(self, model=None):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "SkippHandler"

        self._model = model

    def pre_process_data(self, data):
        self._model.pre_process_data(data)
        self.last_skips = defaultdict(lambda: 0)
        self.skip_after_skip = defaultdict(lambda: 0)
        self.after_skip = defaultdict(lambda: 0)

    def predict(self, student, item, extra=None):
        skip_K = 1
        last_skips = self.last_skips[student]
        if last_skips > 0:
            if self.after_skip[last_skips] == 0:
                skip_K = 0
            else:
                skip_K = self.skip_after_skip[last_skips] / self.after_skip[last_skips]

        return skip_K * self._model.predict(student, item, extra)

    def update(self, student, item, prediction, correct, extra=None):
        skip = type(extra["answer"]) == float and np.isnan(extra["answer"])
        if self.last_skips[student] > 0:
            self.after_skip[self.last_skips[student]] += 1
            if skip:
                self.skip_after_skip[self.last_skips[student]] += 1
        if skip:
            self.last_skips[student] += 1
        else:
            self.last_skips[student] = 0

        self._model.update(student, item, prediction, correct, extra)   # update only when answer is not nan

