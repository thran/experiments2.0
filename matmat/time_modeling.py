from models.eloHierarchical import EloHierarchicalModel
from models.time_models import TimeAvgModel, TimeCombiner
from utils import data
from utils.evaluator import Evaluator
import matplotlib.pylab as plt

data = data.Data("../data/matmat/2016-06-27/answers.pd")
hierarchicalModel = EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)
timeModel = TimeAvgModel()
model = TimeCombiner(hierarchicalModel, timeModel)

Evaluator(data, model).get_report(force_run=True)
# Evaluator(data, model).brier_graphs()
plt.show()
