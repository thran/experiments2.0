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
        self.model_time = True

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
        self.model_time = True

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
        self.model_time = True

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
        self.model_time = True

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


class TimePriorCurrentModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, KC=1, KI=1, init_avg=0, first_level=3):
        Model.__init__(self)
        self.VERSION = 4
        self.name = "Prior-Current-Time"
        self.model_time = True

        self._alpha = alpha
        self._beta = beta
        self._KC = KC
        self._KI = KI
        self._init_avg = init_avg
        self._first_level = first_level

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = self._init_avg
        self.global_skill = defaultdict(lambda: self._log_avg)
        self.local_skill = defaultdict(lambda: defaultdict(lambda: None))
        self.intensity = defaultdict(lambda: 1)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))
        if self._first_level:
            self.item_parents = data.get_item_assignment()

    def predict(self, student, item, extra=None):
        if self._first_level:
            item = self.item_parents[item]
        if self.local_skill[item][student] is None:
            self.local_skill[item][student] = self.global_skill[student]

        prediction = self.intensity[item] - self.local_skill[item][student]
        return math.exp(prediction)

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        if self._first_level:
            item = self.item_parents[item]
        dif = (math.log(time_prediction) - math.log(response_time))

        if self.first_attempt[item][student]:
            self.intensity[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.student_attempts[student] += 1
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False
        K = self._KC if correct else self._KI
        self.local_skill[item][student] += K * dif

    def get_skill(self, student):
        return self.global_skill[student]

    def get_difficulty(self, item):
        return self.intensity[item]

class TimeEloHierarchicalModel(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, alpha=0.8, beta=0.08, KC=0.075, KI=0.1):
        Model.__init__(self)
        self.VERSION = 4
        self.name = "TimeHierarchical"
        self.model_time = True

        self._alpha = alpha
        self._beta = beta
        self._KC = KC
        self._KI = KI

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self._count = 0
        self._log_avg = 0
        self.skill_parents = data.get_skill_structure()
        self.item_parents = data.get_item_assignment()
        self.skill = defaultdict(lambda: defaultdict(lambda: None))
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

        if self.student_attempts[1][student] == 0:
            self._count += 1
            self._log_avg += (dif - self._log_avg) / self._count

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
            s = self.skill[skill][student]
            if s is None:
                s = self._log_avg if skill == 1 else 0
                self.skill[skill][student] = s
            skill_value += s
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

class TimeConcepts(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, alpha=1.0, beta=0.1, K=1, concepts=None):
        Model.__init__(self)
        self.VERSION = 2
        self.name = "TimeConcepts"
        self.model_time = True

        self._alpha = alpha
        self._beta = beta
        self._K = K
        self._concepts = sorted(concepts.keys()) if concepts is not None else "All"
        self._init_concept_map(concepts)

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.global_skill = defaultdict(lambda: 0)
        self.concept_skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.student_attempts = defaultdict(lambda: 0)
        self.student_concept_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.intensity = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        concept = self._get_concept(item)
        if self.concept_skill[concept][student] is None:
            self.concept_skill[concept][student] = self.global_skill[student]

        skill = self.concept_skill[concept][student]
        prediction = self.intensity[item] - skill
        return math.exp(prediction)

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        concept = self._get_concept(item)
        dif = (math.log(time_prediction) - math.log(response_time))

        if self.first_attempt[item][student]:
            self.intensity[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif

        self.concept_skill[concept][student] += self._K * dif
        self.student_concept_attempts[concept][student] += 1
        self.student_attempts[student] += 1

    def _get_concept(self, item):
        if self.concept_map is None:
            return 0
        if item not in self.concept_map:
            return "other"
        return self.concept_map[item]

    def _init_concept_map(self, concepts):
        if concepts is None:
            self.concept_map = None
            return
        self.concept_map = {}
        for concept, items in concepts.items():
            self.concept_map.update({item: concept for item in items})

    def get_difficulties(self, items):
        return [self.intensity[i] for i in items]

    def get_skills(self, students):
        return [self.global_skill[s] for s in students]
