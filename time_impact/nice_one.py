from collections import defaultdict

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

from models.eloConcepts import EloConcepts
from models.eloPriorCurrent import EloPriorCurrentModel
from utils import data as d
from utils.data import LinearDrop, items_in_concept
from utils.runner import Runner, get_hash
from utils.utils import Cache


@Cache(type='json')
def get_difficulties(data, model, items, cache=None):
    Runner(data, model).run(force=True)
    return model.get_difficulties(items)


basic_data = lambda response_modification: d.Data("../data/matmat/2016-12-11/answers.pd",
                                                  response_modification=response_modification,
                                                  # filter_items={'visualization': ['written_question']},
                                                  )
concepts = basic_data(None).get_concepts()
# basic_model = lambda time_tra: EloPriorCurrentModel(KC=2, KI=0.5)
basic_model = lambda time_tra: EloConcepts(concepts=concepts, separate=True)
time_median = 7
min_answers = 20
items = pd.read_csv('../data/matmat/2016-12-11/items.csv', index_col='id')


results = defaultdict(lambda: defaultdict(lambda : {}))
for response_modification in [
    None,
    # TimeLimitResponseModificator([(time_median, 0.5)]),
    LinearDrop(time_median * 2),
]:
    data = basic_data(response_modification)
    model = basic_model(None)
    data.trim_times()
    answers = data.get_dataframe_all()

    for concept in ['numbers', 'multiplication_small']: #['numbers', 'addition', 'subtraction', 'multiplication', 'division']:
        items_ids = list(set(data.get_items()) & set(items_in_concept(data, concept)))
        concept_answers = answers[answers['item'].isin(items_ids)]

        results[concept][str(response_modification)] = (
            items_ids,
            get_difficulties(data, model, items_ids, cache=get_hash(model, data) + concept + '1'),
            concept_answers.groupby('item')['response_time'].median().loc[items_ids],
            concept_answers.groupby('item')['correct'].mean().loc[items_ids]
        )

for concept, r in results.items():
    df_difficulties = pd.DataFrame(index=list(r.values())[0][0])
    for response_modification, (_, difficulties, _, _) in sorted(r.items()):
        df_difficulties[response_modification] = difficulties
    corr_d, count_d = df_difficulties.corr(method='spearman').ix[0, 1], len(list(r.values())[0][2]),

    plt.figure(figsize=(15, 10))
    plt.suptitle(concept.title())
    if 1: # plt
        plt.title('Spearman correlation: {:.3f}'.format(corr_d))
        plt.xlabel('Correctness')
        plt.ylabel('Correctness + Time')
        items_ids, _, rts, cs = results[concept][str(None)]
        marker_index = 0
        for visualization in items['visualization'].unique():
            selected_items = items[items.index.isin(items_ids)]['visualization'] == visualization
            if (selected_items.sum() == 0):
                continue
            plt.scatter(
                df_difficulties.ix[selected_items, 0],
                df_difficulties.ix[selected_items, 1],
                s =rts[selected_items] / rts.max() * 500,
                alpha=0.7,
                cmap='viridis',
                c = cs[selected_items],
                vmin=cs.min(),
                vmax=cs.max(),
                marker=['o', 's', '^', 'p'][marker_index],
                label=visualization,
            )
            marker_index +=1
        plt.legend()
        plt.colorbar()
        if 1:
            for item, x, y in zip(items_ids, df_difficulties.ix[:, 0], df_difficulties.ix[:, 1]):
                plt.text(x, y, items['question'].loc[item], alpha=0.8, fontsize=10)


plt.show()