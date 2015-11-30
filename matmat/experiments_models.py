from utils import data, evaluator
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import AvgModel, ItemAvgModel
from models.eloHierarchical import EloHierarchicalModel
from utils.model_comparison import compare_models
import matplotlib.pylab as plt
import seaborn as sns

d = data.Data("../data/matmat/2015-11-20/answers.pd")

compare_models(d, [
    AvgModel(),
    ItemAvgModel(),
    EloPriorCurrentModel(),
    EloHierarchicalModel(),
    EloHierarchicalModel(alpha=0.25, beta=0.02),
], dont=0, force_evaluate=0, answer_filters={
    "long (50) student": data.filter_students_with_many_answers(),
    "long (30) student": data.filter_students_with_many_answers(number_of_answers=30),
    "long (11) student": data.filter_students_with_many_answers(number_of_answers=11),
})


# evaluator.Evaluator(d, EloHierarchicalModel(alpha=0.25, beta=0.02)).brier_graphs()
# evaluator.Evaluator(d, EloPriorCurrentModel()).brier_graphs()
# evaluator.Evaluator(d, ItemAvgModel()).brier_graphs()




plt.show()
