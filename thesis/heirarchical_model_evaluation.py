import json

from models.eloBasic import EloBasicModel
from models.eloPriorCurrent import EloPriorCurrentModel
from utils import evaluator, utils
from models.eloConcepts import EloConceptsOld, EloConcepts
from models.eloNetwork import EloNetwork
from models.model import AvgModel, ItemAvgModel, StudentAvgModel
from models.eloHierarchical import EloHierarchicalModel
from utils.data import Data
from utils.model_comparison import compare_models
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

from utils.runner import Runner

data = Data('data/matmat/answers.pd', train_size=0.3, only_first=False, filter=(100, 10))
concepts = data.get_concepts()

compare_models(data, [
    AvgModel(),
    ItemAvgModel(),
    StudentAvgModel(),
    EloPriorCurrentModel(KC=2, KI=0.5),
    EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
    EloHierarchicalModel(KC=1, KI=0.75, alpha=1.2, beta=0.04),
], dont=0, force_evaluate=0, force_run=0, )


plt.show()