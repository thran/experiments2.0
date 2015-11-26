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
], dont=1)


evaluator.Evaluator(d, EloHierarchicalModel(alpha=0.25, beta=0.02)).brier_graphs()
evaluator.Evaluator(d, EloPriorCurrentModel()).brier_graphs()
evaluator.Evaluator(d, ItemAvgModel()).brier_graphs()




plt.show()
