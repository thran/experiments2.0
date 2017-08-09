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


if 0:
    data = Data('data/anatom/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
    # data = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True)
    concepts = json.load(open('data/anatom/concepts_combinations.json'))
    concepts_locations = json.load(open('data/anatom/concepts_locations.json'))
    concepts_systems = json.load(open('data/anatom/concepts_systems.json'))
    print(len(concepts))
    print(len(concepts_locations))
    print(len(concepts_systems))
    compare_models(data, [
        AvgModel(),
        ItemAvgModel(),
        StudentAvgModel(),
        EloBasicModel(alpha=1., beta=0.04),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_locations),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_systems),
        EloNetwork(alpha=1, beta=0.04, w1=0.8, w2=0.6),
    ], dont=0, force_evaluate=0, force_run=0, diff_to=0.414416)

    if 0:
        utils.grid_search(data, EloConceptsOld,
                          {"alpha": 1., "beta": 0.04, 'concepts': concepts}, {
            # "alpha": np.arange(0.4, 2, 0.2),
            # "beta": np.arange(0.0, 0.2, 0.02),
            'gamma': np.arange(0, .6, 0.2)
        # }, plot_axes=["beta", "alpha"])
        }, plot_axes="gamma")

if 1:
    data = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
    # data = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True)
    concepts = json.load(open('data/slepemapy/concepts_combinations.json'))
    concepts_locations = json.load(open('data/slepemapy/concepts_locations.json'))
    concepts_types = json.load(open('data/slepemapy/concepts_types.json'))
    compare_models(data, [
        AvgModel(),
        ItemAvgModel(),
        StudentAvgModel(),
        EloBasicModel(alpha=1., beta=0.04),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts),
        # EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_locations),
        # EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_types),
        # EloNetwork(alpha=1.2, beta=0.04, w1=.6, w2=0.6),
        EloNetwork(alpha=1, beta=0.04, w1=.8, w2=0.6),
    ], dont=1, force_evaluate=0, force_run=0, )

    if 0:
        utils.grid_search(data, EloPriorCurrentModel,
                          {"KC": 1, "KI": 1}, {
                          # {"alpha": 0.25, "beta": 0.02}, {
            "alpha": np.arange(0.2, 2, 0.2),
            "beta": np.arange(0.02, 0.2, 0.02),
            # "KC": np.arange(1.5, 5.0, 0.25),
            # "KI": np.arange(1.25, 4.5, 0.25),
        # }, plot_axes=["KC", "KI"])
        }, plot_axes=["beta", "alpha"])

    if 0:
        concepts_5 = json.load(open('data/slepemapy/concepts_5.json'))
        concepts_20 = json.load(open('data/slepemapy/concepts_20.json'))
        concepts_50 = json.load(open('data/slepemapy/concepts_50.json'))
        utils.grid_search(data, EloConceptsOld,
                          {"alpha": 1., "beta": 0.04, 'concepts': concepts_5}, {
            # "alpha": np.arange(0.4, 2, 0.2),
            # "beta": np.arange(0.0, 0.2, 0.02),
            'gamma': np.arange(0, .6, 0.2)
        # }, plot_axes=["beta", "alpha"])
        }, plot_axes="gamma")

    if 1:
        utils.grid_search(data, EloNetwork,
                          {"alpha": 1.2, "beta": 0.04}, {
            'w1': np.arange(0.4, 1.3, 0.2),
            'w2': np.arange(0.4, 1.3, 0.2),
        }, plot_axes=['w1', 'w2'])

if 0:
    data = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
    concepts = json.load(open('data/slepemapy/concepts_combinations.json'))
    concepts_locations = json.load(open('data/slepemapy/concepts_locations.json'))
    concepts_types = json.load(open('data/slepemapy/concepts_types.json'))
    concepts_corrected = json.load(open('data/slepemapy/concepts_combinations_corrected.json'))
    concepts_locations_corrected = json.load(open('data/slepemapy/concepts_locations_corrected.json'))
    concepts_types_corrected = json.load(open('data/slepemapy/concepts_types_corrected.json'))
    concepts_5 = json.load(open('data/slepemapy/concepts_5.json'))
    concepts_20 = json.load(open('data/slepemapy/concepts_20.json'))
    concepts_50 = json.load(open('data/slepemapy/concepts_50.json'))
    compare_models(data, [
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts_locations),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts_types),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts_corrected),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts_locations_corrected),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts_types_corrected),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_5),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_20),
        EloConceptsOld(alpha=1., beta=0.04, gamma=0.2, concepts=concepts_50),
    ], dont=0, force_evaluate=0, force_run=0, diff_to=0.41345)


# if 1:
#     europe = json.load(open('data/slepemapy/Europe.json'))
#     data = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True, filter=(100, 10), filter_items={'id': map(int, list(europe.keys()))})
#     compare_models(data, [
#         EloBasicModel(alpha=1.2, beta=0.04),
#         EloConceptsOld(alpha=1.2, beta=0.04, gamma=0.5, concepts=concepts),
#     ], dont=0, force_evaluate=0, force_run=0, )

if 0:
    data = Data('data/matmat/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
    concepts = data.get_concepts()

    compare_models(data, [
        AvgModel(),
        ItemAvgModel(),
        StudentAvgModel(),
        EloBasicModel(alpha=1.2, beta=0.04),
        EloConceptsOld(alpha=1.2, beta=0.04, gamma=0.5, concepts=concepts),
        EloNetwork(alpha=1.2, beta=0.04, w1=1, w2=1.5),
        EloNetwork(alpha=1.2, beta=0.04, w1=0.6, w2=0.6),
    ], dont=1, force_evaluate=0, force_run=0, )

    if 0:
        utils.grid_search(data, EloConceptsOld,
                          {"alpha": 1.2, "beta": 0.04, 'concepts': concepts}, {
            # "alpha": np.arange(0.4, 2, 0.2),
            # "beta": np.arange(0.0, 0.2, 0.02),
            'gamma': np.arange(0, 1., 0.1)
        # }, plot_axes=["beta", "alpha"])
        }, plot_axes="gamma")

    if 0:
        utils.grid_search(data, EloNetwork,
                          {"alpha": 1.2, "beta": 0.04}, {
            'w1': np.arange(0.8, 1.3, 0.2),
            'w2': np.arange(0.8, 1.3, 0.2),
        }, plot_axes=['w1', 'w2'])


plt.show()