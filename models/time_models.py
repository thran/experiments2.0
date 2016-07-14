from collections import defaultdict

import math

from models.model import Model


class TimeCombiner(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, predictionModel, timeModel):
        Model.__init__(self)
        self._predictionModel = predictionModel
        self._timeModel = timeModel

    def __str__(self):
        return "{} + {}".format(self._predictionModel, self._timeModel)

    def pre_process_data(self, data):
        self._predictionModel.pre_process_data(data)
        self._timeModel.pre_process_data(data)

    def post_process_data(self, data):
        self._predictionModel.post_process_data(data)
        self._timeModel.post_process_data(data)

    def predict(self, student, item, extra):
        prediction = self._predictionModel.predict(student, item, extra)
        time_prediction = self._timeModel.predict(student, item, extra)
        return prediction, time_prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        self._predictionModel.update(student, item, prediction, time_prediction, correct, response_time, extra)
        self._timeModel.update(student, item, prediction, time_prediction, correct, response_time, extra)


class TimeAvgModel(Model):
    def __init__(self, init_avg=1):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "TimeGlobal-average"

        self._init_avg = init_avg

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = self._init_avg

    def predict(self, student, item, extra=None):
        return math.exp(self._log_avg)

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self._count += 1
        self._log_avg += (math.log(response_time) - self._log_avg) / self._count


class TimeStudentAvgModel(Model):
    def __init__(self, init_avg=1):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "TimeStudent-average"

        self._init_avg = init_avg

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = self._init_avg
        self._counts = defaultdict(lambda: self._log_avg)
        self._log_avgs = defaultdict(lambda: self._log_avg)

    def predict(self, student, item, extra=None):
        return math.exp(self._log_avgs[student])

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self._count += 1
        self._log_avg += (math.log(response_time) - self._log_avg) / self._count
        self._counts[student] += 1
        self._log_avgs[student] += (math.log(response_time) - self._log_avgs[student]) / self._counts[student]



class TimeItemAvgModel(Model):
    def __init__(self, init_avg=1):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "TimeItem-average"

        self._init_avg = init_avg

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = self._init_avg
        self._counts = defaultdict(lambda: self._log_avg)
        self._log_avgs = defaultdict(lambda: self._log_avg)

    def predict(self, student, item, extra=None):
        return math.exp(self._log_avgs[item])

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        self._count += 1
        self._log_avg += (math.log(response_time) - self._log_avg) / self._count
        self._counts[item] += 1
        self._log_avgs[item] += (math.log(response_time) - self._log_avgs[item]) / self._counts[item]


class BasicTimeModel(Model):
    def __init__(self, alpha=1.0, beta=0.1, K=1):
        Model.__init__(self)
        self.VERSION = 2
        self.name = "Basic-Time"

        self._alpha = alpha
        self._beta = beta
        self._K = K

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.skill = defaultdict(lambda: 0)
        self.intensity = defaultdict(lambda: 1)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        prediction = self.intensity[item] - self.skill[student]
        return math.exp(prediction)

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        dif = (math.log(time_prediction) - math.log(response_time))

        if self.first_attempt[item][student]:
            self.intensity[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False

        self.skill[student] += self._K * dif
        self.student_attempts[student] += 1
