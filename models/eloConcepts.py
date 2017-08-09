import json
from collections import defaultdict
from models.model import Model


class EloConcepts(Model):
    """
    """

    def __init__(self, alpha=1.0, beta=0.1, concepts=None, separate=False):
        Model.__init__(self)
        self.VERSION = 3
        self.name = "Concepts"

        self._alpha = alpha
        self._beta = beta
        self._separate = separate
        self._concepts = sorted(concepts.keys()) if concepts is not None else "All"
        self._init_concept_map(concepts)

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.global_skill = defaultdict(lambda: 0)
        if self._separate:
            self.concept_skill = defaultdict(lambda: defaultdict(lambda: None))
        else:
            self.concept_skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.student_attempts = defaultdict(lambda: 0)
        self.student_concept_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.difficulty = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        concept = self._get_concept(item)
        if self.concept_skill[concept][student] is None:
            self.concept_skill[concept][student] = self.global_skill[student]

        skill = self.concept_skill[concept][student]
        random_factor = 0
        if 'options' in extra:
            options = len(json.loads(extra['options']))
            if options > 0:
                random_factor = 1. /options
        prediction = self._sigmoid(skill - self.difficulty[item], random_factor)

        return prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        concept = self._get_concept(item)
        dif = (correct - prediction)

        if self.first_attempt[item][student]:
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.concept_skill[concept][student] += self.decay_function(self.student_concept_attempts[concept][student]) * dif
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.student_concept_attempts[concept][student] += 1
            self.student_attempts[student] += 1
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False

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
        return [self.difficulty[i] for i in items]

    def get_skill(self, student, concept=None):
        if concept is None:
            return self.global_skill[student]
        return self.concept_skill[concept][student]



class EloConceptsOld(Model):
    """
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=1, concepts=None):
        Model.__init__(self)
        self.VERSION = 3
        self.name = "ConceptsOld"

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._concepts = sorted(concepts.keys()) if concepts is not None else "All"
        self._init_concept_map(concepts)

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.global_skill = defaultdict(lambda: 0)
        self.concept_skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.student_attempts = defaultdict(lambda: 0)
        self.student_concept_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.difficulty = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        concept = self._get_concept(item)
        skill = self.global_skill[student] + self.concept_skill[concept][student]
        prediction = self._sigmoid(skill - self.difficulty[item], self.get_random_factor(extra))

        return prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra):
        concept = self._get_concept(item)
        dif = (correct - prediction)

        if self.first_attempt[item][student]:
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.concept_skill[concept][student] += self._gamma * self.decay_function(self.student_concept_attempts[concept][student]) * dif
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
            self.student_concept_attempts[concept][student] += 1
            self.student_attempts[student] += 1
            self.item_attempts[item] += 1
            self.first_attempt[item][student] = False

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
        return [self.difficulty[i] for i in items]

    def get_skill(self, student, concept=None):
        if concept is None:
            return self.global_skill[student]
        return self.concept_skill[concept][student] + self.global_skill[student]