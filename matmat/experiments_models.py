from utils import data, evaluator
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import AvgModel, ItemAvgModel
from utils.model_copariosn import compare_models
import matplotlib.pylab as plt
import seaborn as sns

data = data.Data("../data/matmat/2015-11-20/answers.pd")

compare_models(data, [
    AvgModel(),
    ItemAvgModel(),
    EloPriorCurrentModel(),
], dont=0)

plt.show()