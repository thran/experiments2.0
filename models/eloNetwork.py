import json
from collections import defaultdict

import numpy as np
import pandas as pd

from cross_system.clustering.similarity import similarity_pearson
from models.model import Model
from utils.utils import Cache


@Cache(hash=True)
def get_similarity(data, cache=None):
    print('Computing similarities ' + cache)
    return similarity_pearson(similarity_pearson(data.get_dataframe_train(), min_periods=200))
    # return similarity_pearson(data.get_dataframe_train())

class EloNetwork(Model):
    """
    """

    def __init__(self, alpha=1.0, beta=0.1, w1=1., w2=1.):
        Model.__init__(self)
        self.VERSION = 6
        self.name = "Network"

        self._alpha = alpha
        self._beta = beta
        self._w1 = w1
        self._w2 = w2


        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):

        self.similarities = get_similarity(data, cache='corrs3 ' + str(data))
        self.items = list(self.similarities.index)

        self.global_skill = defaultdict(lambda: 0)
        self.item_skill = defaultdict(lambda: np.zeros(len(self.items)))
        self.student_attempts = defaultdict(lambda: 0)
        self.difficulty = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        skill = self._w1 * self.global_skill[student] + self._w2 * int(self.item_skill[student][self.items.index(item)])
        prediction = self._sigmoid(skill - self.difficulty[item], self.get_random_factor(extra))

        return prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        dif = (correct - prediction)

        if self.first_attempt[item][student]:
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.item_skill[student] += self.similarities[item].values * dif
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.student_attempts[student] += 1
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False

    def get_difficulties(self, items):
        return [self.difficulty[i] for i in items]

    def get_skill(self, student, concept=None):
        return self.global_skill[student]