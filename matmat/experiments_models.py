from models.skipHandler import SkipHandler
from utils import data, evaluator, utils
from models.eloPriorCurrent import EloPriorCurrentModel
from models.eloConcepts import EloConcepts
from models.model import AvgModel, ItemAvgModel
from models.eloHierarchical import EloHierarchicalModel
from utils.model_comparison import compare_models
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

d = data.Data("../data/matmat/2015-11-20/answers.pd", train_size=0.5, only_first=True)
# d = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.5)]))
# d = data.Data("../data/matmat/2015-11-20/answers.pd", response_modification=data.TimeLimitResponseModificator([(5, 0.66), (10, 0.33)]))
concepts = d.get_concepts()
compare_models(d, [
    # AvgModel(),
    ItemAvgModel(),
    SkipHandler(ItemAvgModel()),
    # EloPriorCurrentModel(),
    EloPriorCurrentModel(KC=2, KI=0.5),
    SkipHandler(EloPriorCurrentModel(KC=2, KI=0.5)),
    # EloHierarchicalModel(),
    # EloHierarchicalModel(KC=1, KI=0.75),
    EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
    SkipHandler(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02)),
    # EloHierarchicalModel(alpha=0.25, beta=0.02),
    # EloConcepts(),
    EloConcepts(concepts=concepts),
    SkipHandler(EloConcepts(concepts=concepts)),
], dont=0, force_evaluate=0, force_run=0, runs=20, answer_filters={
    # "long (50) student": data.filter_students_with_many_answers(),
    # "long (30) student": data.filter_students_with_many_answers(number_of_answers=30),
    # "long (11) student": data.filter_students_with_many_answers(number_of_answers=11),
    # "response >5s-0.5": data.transform_response_by_time(((5, 0.5),))
})



# evaluator.Evaluator(d, EloHierarchicalModel(alpha=0.25, beta=0.02)).brier_graphs()
# evaluator.Evaluator(d, EloPriorCurrentModel()).brier_graphs()
# evaluator.Evaluator(d, ItemAvgModel()).brier_graphs()

if 0:
    utils.grid_search(d, EloHierarchicalModel,
                      # {"KC": 1, "KI": 0.75}, {
                      {"alpha": 0.25, "beta": 0.02}, {
        # "alpha": np.arange(0.2, 1.3, 0.2),
        # "beta": np.arange(0., 0.2, 0.02),
        "KC": np.arange(1.5, 5.0, 0.25),
        "KI": np.arange(1.25, 4.5, 0.25),
    }, plot_axes=["KC", "KI"])
    # }, plot_axes=["alpha", "beta"])

plt.show()
