import json
from collections import defaultdict

import pandas as pd

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

dataS = Data('data/slepemapy/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
dataM = Data('data/matmat/answers.pd', train_size=0.3, only_first=True, filter=(100, 10))
concepts = json.load(open('data/slepemapy/concepts_combinations.json'))

if 0:
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    items = dataS.get_items()
    students = dataS.get_students()[:1000]
    model1 = EloBasicModel(alpha=1., beta=0.04)
    model2 = EloConceptsOld(alpha=1., beta=0.04, gamma=0.4, concepts=concepts)
    Runner(dataS, model1).run(True)
    df['Basic model - difficulties'] = model1.get_difficulties(items)
    df2['Basic model - skills'] = model1.get_skills(students)
    Runner(dataS, model2).run(True)
    df['Concept model - difficulties'] = model2.get_difficulties(items)
    df2['Concept model - skills'] = model2.get_skills(students)

    g = sns.jointplot("Basic model - difficulties", "Concept model - difficulties", data=df, kind="reg")
    g = sns.jointplot("Basic model - skills", "Concept model - skills", data=df2, kind="reg")


if 0:
    model = EloBasicModel(alpha=1., beta=0.04)
    data = defaultdict(lambda: [])
    items = {
        159: 'Germany',
        93: 'Finland',
        98: 'Ghana',
        136: 'Kyrgyzstan',
     }
    def callback(answer):
        if answer['item'] in items.keys():
            data[items[answer['item']]].append(model.get_difficulties([answer['item']])[0])

    model.after_update_callback = callback
    Runner(dataS, model).run(True)

    json.dump(data, open('difficulties.json', 'w'))
    print(data)

if 1:
    data = json.load(open('difficulties.json'))
    size = 3000
    for item, values in data.items():
        plt.plot(list(range(min(len(values),size))), values[:size], label=item)
    plt.plot(list(range(size)), [1 / (1 + 0.04 * n) for n in range(size)])

    plt.legend()

plt.show()
