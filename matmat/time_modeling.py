from models.eloHierarchical import EloHierarchicalModel
from models.model import ItemAvgModel, StudentAvgModel, AvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel
from utils import data
from utils.evaluator import Evaluator
import matplotlib.pylab as plt

from utils.model_comparison import compare_models

data = data.Data("../data/matmat/2016-06-27/answers.pd")
data.trim_times()


compare_models(data, [
    TimeCombiner(AvgModel(), TimeAvgModel()),
    TimeCombiner(ItemAvgModel(), TimeItemAvgModel()),
    TimeCombiner(StudentAvgModel(), TimeStudentAvgModel()),
    TimeCombiner(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), TimeStudentAvgModel()),
])


model = TimeCombiner(ItemAvgModel(), TimeStudentAvgModel())
# Evaluator(data, model).get_report(force_run=True)
# Evaluator(data, model).brier_graphs()
# Evaluator(data, model).brier_graphs(time=True)
# Evaluator(data, model).brier_graphs(time_raw=True)
plt.show()
