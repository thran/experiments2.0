from collections import defaultdict
from models.model import Model


class EloHierarchicalModel(Model):
    """
    Model work with tree structure and update all ancestors with level based decay
    """

    def __init__(self, alpha=1.0, beta=0.1, KC=3.5, KI=2.5):
        Model.__init__(self)
        self.VERSION = 1
        self.name = "Hierarchical"

        self._alpha = alpha
        self._beta = beta
        self._KC = KC
        self._KI = KI

        self.decay_function = lambda x: alpha / (1 + beta * x)

    def pre_process_data(self, data):
        self.skill_parents = data.get_skill_structure()
        self.item_parents = data.get_item_assignment()
        self.skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def predict(self, student, item, extra=None):
        skill = self._get_skill(student, self.item_parents[item])
        prediction = self._sigmoid(skill - self.difficulty[item])

        return prediction

    def update(self, student, item, prediction, time_prediction, correct, response_time, extra=None):
        dif = (correct - prediction)

        for level, skill in enumerate(self._get_parents(item)):
            p = self._sigmoid(self._get_skill(student, skill) - self.difficulty[item])
            K = self._KC if correct else self._KI
            decay = self.decay_function(self.student_attempts[skill][student])
            self.skill[skill][student] += decay * (correct - p) * K
            self.student_attempts[skill][student] += 1

        if self.first_attempt[item][student]:
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif

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

    def get_skills(self, students):
        return [self.skill[1][s] for s in students]

    def get_difficulties(self, items):
        return [self.difficulty[i] for i in items]
