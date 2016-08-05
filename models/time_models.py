from collections import defaultdict

import math

from models.model import Model


class TimeCombiner(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, prediction_model, time_model):
        Model.__init__(self)
        self._prediction_model = prediction_model
        self._time_model = time_model

    def __str__(self):
        return "{} + {}".format(self._prediction_model, self._time_model)

    def pre_process_data(self, data):
        self._prediction_model.pre_process_data(data)
        self._time_model.pre_process_data(data)

    def post_process_data(self, data):
        self._prediction_model.post_process_data(data)
        self._time_model.post_process_data(data)

    def predict(self, student, item, extra):
        prediction = self._prediction_model.predict(student, item, extra)
        time_prediction = self._time_model.predict(student, item, extra)
        return prediction, time_prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        self._prediction_model.update(student, item, prediction, time_prediction, correct, response_time, extra)
        self._time_model.update(student, item, prediction, time_prediction, correct, response_time, extra)


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

    def get_skill(self, student):
        return - self._log_avgs[student]


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

    def get_difficulty(self, item):
        return self._log_avgs[item]


class BasicTimeModel(Model):
    def __init__(self, alpha=1.0, beta=0.1, K=1, init_avg=0, floating_start=True):
        Model.__init__(self)
        self.VERSION = 3
        self.name = "Basic-Time"

        self._alpha = alpha
        self._beta = beta
        self._K = K
        self._init_avg = init_avg
        self._floating_start = floating_start

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = self._init_avg
        self.skill = defaultdict(lambda: self._log_avg)
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

        if self.student_attempts[student] == 0 and  self._floating_start:
            self._count += 1
            self._log_avg += (dif - self._log_avg) / self._count

        self.skill[student] += self._K * dif
        self.student_attempts[student] += 1

    def get_skill(self, student):
        return self.skill[student]

    def get_difficulty(self, item):
        return self.intensity[item]


class TimeEloHierarchicalModel(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, alpha=0.8, beta=0.08, KC=0.075, KI=0.1):
        Model.__init__(self)
        self.VERSION = 2
        self.name = "TimeHierarchical"

        self._alpha = alpha
        self._beta = beta
        self._KC = KC
        self._KI = KI

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.skill_parents = data.get_skill_structure()
        self.item_parents = data.get_item_assignment()
        self.skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.intensity = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        skill = self._get_skill(student, self.item_parents[item])
        prediction = self.intensity[item] - skill
        return math.exp(prediction)

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        dif = (math.log(time_prediction) - math.log(response_time))

        for level, skill in enumerate(self._get_parents(item)):
            p = self.intensity[item] - self._get_skill(student, skill)
            dif = (p - math.log(response_time))
            K = self._KC if correct else self._KI
            decay = self.decay_function(self.student_attempts[skill][student])
            self.skill[skill][student] += decay * dif * K
            self.student_attempts[skill][student] += 1

        if self.first_attempt[item][student]:
            self.intensity[item] -= self.decay_function(self.item_attempts[item]) * dif

        self.item_attempts[item] += 1
        self.first_attempt[item][student] = False

    def _get_skill(self, student, skill):
        skill_value = 0
        while skill is not None:
            skill_value += self.skill[skill][student]
            skill = self.skill_parents[skill]
        return skill_value

    def _get_parents(self, item):
        skills = []
        skill = self.item_parents[item]
        while skill is not None:
            skills.append(skill)
            skill = self.skill_parents[skill]
        return skills[::-1]

    def get_skill(self, student):
        return self.skill[1][student]

    def get_difficulty(self, item):
        return self.intensity[item]
