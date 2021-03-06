import json
from collections import defaultdict
from models.model import Model


class EloPriorCurrentModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, KC=1, KI=1):
        Model.__init__(self)
        self.VERSION = 4
        self.name = "Prior-current"

        self._alpha = alpha
        self._beta = beta
        self._KC = KC
        self._KI = KI

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.global_skill = defaultdict(lambda: 0)
        self.local_skill = defaultdict(lambda: defaultdict(lambda: None))
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        if self.local_skill[item][student] is None:
            self.local_skill[item][student] = self.global_skill[student]

        prediction = self._sigmoid(self.local_skill[item][student] - self.difficulty[item], self.get_random_factor(extra))
        return prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        dif = (correct - prediction)

        if self.first_attempt[item][student]:
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.student_attempts[student] += 1
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False
        K = self._KC if correct else self._KI
        self.local_skill[item][student] += K * dif

    def get_skills(self, students):
        return [self.global_skill[s] for s in students]

    def get_difficulties(self, items):
        return [self.difficulty[i] for i in items]
