import numpy as np
import pandas as pd
from utils import data as d
from models.eloHierarchical import EloHierarchicalModel
from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import ItemAvgModel, StudentAvgModel, AvgModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel, \
    TimeEloHierarchicalModel, TimePriorCurrentModel
from utils.model_comparison import compare_models
import matplotlib.pylab as plt
import seaborn as sns

from utils.utils import grid_search


class MathGardenData(d.Data):
    def get_item_assignment(self, filename="items.csv"):
        items = self.get_items()
        return {i: i for i in items}

    def get_skill_structure(self, filename="skills.csv"):
        items = self.get_items()
        s = {i: 0 for i in items}
        s[0] = None
        return s


# data_all = MathGardenData('../data/mathgarden/multiplication.pd')
data_all = MathGardenData('../data/mathgarden/addition.pd')
# data_all = MathGardenData('../data/mathgarden/subtraction.pd')
data = MathGardenData('../data/mathgarden/multiplication.pd', test_subset=100000)
data.trim_times()
data_all.trim_times()


def grid_search_K():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), BasicTimeModel(**kwargs)),
                {"alpha": 0.6, "beta": 0.1},
                {"K": np.arange(0, 1, 0.05)},
                plot_axes='K', time=True,
                )



def grid_search_AB():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), BasicTimeModel(**kwargs)),
                {"K": 0.2}, {
                    "alpha": np.arange(0.4, 1.3, 0.2),
                    "beta": np.arange(0.06, 0.2, 0.02),
                }, plot_axes=['alpha', 'beta'], time=True,
                )


def grid_search_Ks():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimePriorCurrentModel(**kwargs)),
                {"alpha": 0.6, "beta": 0.1},
                {"KC": np.arange(0.1, 0.7, 0.1),"KI": np.arange(0.1, 0.7, 0.1)},
                plot_axes=['KI', 'KC'], time=True,
            )

def grid_search_AB2():
    grid_search(data, lambda **kwargs: TimeCombiner(AvgModel(), TimePriorCurrentModel(**kwargs)),
                {"KI": 0.3, 'KC': 0.4}, {
                    "alpha": np.arange(0.1, 0.5, 0.1),
                    "beta": np.arange(0.02, 0.2, 0.02),
                }, plot_axes=['alpha', 'beta'], time=True,
            )


def item_answers_hist():
    df = data.get_dataframe_all()
    counts = df.groupby(['item', 'student']).apply(len)
    counts[counts > 20] = 20
    sns.distplot(counts)


compare_models(data_all, [
    TimeCombiner(AvgModel(), TimeAvgModel()),
    TimeCombiner(ItemAvgModel(), TimeItemAvgModel()),
    TimeCombiner(StudentAvgModel(), TimeStudentAvgModel()),
    TimeCombiner(EloPriorCurrentModel(KC=2, KI=0.5), BasicTimeModel(alpha=0.6, beta=0.1, K=0.25)),
    TimeCombiner(EloPriorCurrentModel(KC=2, KI=0.5), BasicTimeModel(alpha=0.6, beta=0.18, K=0.2)),
    TimeCombiner(EloPriorCurrentModel(KC=2, KI=0.5), TimePriorCurrentModel(alpha=0.2, beta=0.08, KC=0.4, KI=0.3, first_level=False)),
    # TimeCombiner(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), TimeEloHierarchicalModel()),
    TimeCombiner(EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02), TimeEloHierarchicalModel(alpha=0.8, beta=0.08, KC=0.275, KI=0.225)),
], dont=0)

# grid_search_Ks()
# grid_search_K()
# grid_search_AB()
# grid_search_AB2()

plt.show()
