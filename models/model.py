import inspect

import math
import pandas as pd


class Model:
    def __init__(self):
        self.VERSION = 0
        self.name = "Model"

    def __str__(self):
        s = "{}.{}".format(self.name, self.VERSION)
        (args, _, _, defaults) = inspect.getargspec(self.__init__)
        s += "".join([", {}:{}".format(a, getattr(self, "_" + a)) for a, d in zip(args[-len(defaults):], defaults) if getattr(self, "_" + a) != d])
        return s

    def _sigmoid(self, x, c=0):
        return c + (1 - c) / (1 + math.exp(-x))

    def pre_process_data(self, data):
        pass

    def process_data(self, data, logger=None):
        print("Processing {} on {}".format(self, data))
        print("  training")
        for answer in data.iter_train():
            prediction = self.predict(answer["student"], answer["item"], answer)
            self.update(answer["student"], answer["item"], prediction, answer["correct"], answer)

        print("  testing")
        for answer in data.iter_test():
            prediction = self.predict(answer["student"], answer["item"], answer)
            self.update(answer["student"], answer["item"], prediction, answer["correct"], answer)
            if logger is not None:
                logger(answer, prediction)

    def predict(self, student, item, extra):
        pass

    def update(self, student, item, prediction, correct, extra):
        pass


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

    def update(self, student, item, prediction, correct, extra=None):
        self._all += 1
        if correct:
            self._corrects += 1
        self._avg = float(self._corrects) / self._all


class ItemAvgModel(Model):
    def __init__(self, init_avg=0.5):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "Item-average"
        self._init_avg = init_avg

    def pre_process_data(self, data):
        items = data.get_items()
        self.corrects = pd.Series(index=items)
        self.counts = pd.Series(index=items)
        self.corrects[:] = 0
        self.counts[:] = 0

    def predict(self, student, item, extra=None):
        return self.corrects[item] / self.counts[item] if self.counts[item] > 0 else self._init_avg

    def update(self, student, item, prediction, correct, extra=None):
        self.counts[item] += 1
        if correct:
            self.corrects[item] += 1
