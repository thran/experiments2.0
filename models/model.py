import inspect

import math
from collections import defaultdict

import pandas as pd


class Model:
    def __init__(self):
        self.VERSION = 0
        self.name = "Model"
        self.after_update_callback = None

    def __str__(self):
        if self.name == "Model":
            raise AttributeError("Model name not specified")
        if self.VERSION == 0:
            raise AttributeError("Model version not specified")
        s = "{}.{}".format(self.name, self.VERSION)
        (args, _, _, defaults) = inspect.getargspec(self.__init__)
        s += "".join([", {}:{}".format(a, str(getattr(self, "_" + a))) for a, d in zip(args[-len(defaults):], defaults) if getattr(self, "_" + a) != d])
        return s

    def _sigmoid(self, x, c=0):
        return c + (1 - c) / (1 + math.exp(-x))

    def pre_process_data(self, data):
        pass

    def post_process_data(self, data):
        pass

    def process_data(self, data, logger=None, only_train=False):
        def add_time_prediction_if_missing(param):
            if type(param) is not tuple or len(param) == 1:
                return param, None
            return param

        print("Processing {} on {}".format(self, data))
        print("  training")
        for answer in data.iter_train():
            prediction, time_prediction = add_time_prediction_if_missing(self.predict(answer["student"], answer["item"], answer))
            self.update(answer["student"], answer["item"], prediction, time_prediction, answer["correct"], answer["response_time"], answer)
            if self.after_update_callback is not None:
                self.after_update_callback(answer["student"], answer["item"])

        if not only_train:
            print("  testing")
            for answer in data.iter_test():
                prediction, time_prediction = add_time_prediction_if_missing(self.predict(answer["student"], answer["item"], answer))
                self.update(answer["student"], answer["item"], prediction, time_prediction, answer["correct"], answer["response_time"], answer)
                if logger is not None:
                    logger(answer, prediction, time_prediction)
                if self.after_update_callback is not None:
                    self.after_update_callback(answer["student"], answer["item"])

        self.post_process_data(data)

    def predict(self, student, item, extra):
        pass

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        pass

    def get_skills(self, students):
        return [self.get_skill(s) for s in students]

    def get_difficulties(self, items):
        return [self.get_difficulty(i) for i in items]

    def get_skill(self, student):
        return None

    def get_difficulty(self, item):
        return None


class AvgModel(Model):
    def __init__(self, init_avg=0.5):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "Global-average"

        self._init_avg = init_avg

    def pre_process_data(self, data):
        self._corrects = 0
        self._all = 0
        self._avg = self._init_avg

    def predict(self, student, item, extra=None):
        return self._avg

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self._all += 1
        if correct:
            self._corrects += 1
        self._avg = float(self._corrects) / self._all


class ItemAvgModel(Model):
    def __init__(self, init_avg=0.5):
        Model.__init__(self)
        self.VERSION = 2
        self.name = "Item-average"
        self._init_avg = init_avg

    def pre_process_data(self, data):
        self.corrects = defaultdict(lambda: 0)
        self.counts = defaultdict(lambda: 0)
        self._corrects = 0
        self._all = 0
        self._avg = self._init_avg

    def predict(self, student, item, extra=None):
        return self.corrects[item] / self.counts[item] if self.counts[item] > 0 else self._avg

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self.counts[item] += 1
        self.corrects[item] += correct
        self._corrects += correct
        self._all += 1
        self._avg = self._corrects / self._all

    def post_process_data(self, data):
        self.difficulty = pd.Series(self.corrects) / pd.Series(self.counts)

    def get_difficulty(self, item):
        return 1 - (self.corrects[item] / self.counts[item] if self.counts[item] > 0 else self._avg)


class StudentAvgModel(Model):
    def __init__(self, init_avg=0.5):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "Student-average"
        self._init_avg = init_avg

    def pre_process_data(self, data):
        self.corrects = defaultdict(lambda: 0)
        self.counts = defaultdict(lambda: 0)
        self._corrects = 0
        self._all = 0
        self._avg = self._init_avg

    def predict(self, student, item, extra=None):
        return self.corrects[student] / self.counts[student] if self.counts[student] > 0 else self._avg

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self.counts[student] += 1
        self.corrects[student] += correct
        self._corrects += correct
        self._all += 1
        self._avg = self._corrects / self._all

    def post_process_data(self, data):
        self.difficulty = pd.Series(self.corrects) / pd.Series(self.counts)

        def get_skill(self, student):
            return self.corrects[student] / self.counts[student] if self.counts[student] > 0 else self._avg
