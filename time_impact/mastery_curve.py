import os

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import pearsonr

from models.eloPriorCurrent import EloPriorCurrentModel
from models.model import ItemAvgModel, Model
from models.time_models import TimePriorCurrentModel
from time_impact.time_utils import sigmoid, get_difficulties, rolling_success
from utils import data as d
from utils.data import LinearDrop, TimeLimitResponseModificator, \
    items_in_concept
from utils.runner import Runner


def master_curves(answers, metrics, min_answers=50, student_count=None, smooth=0):
    sessions = answers.groupby('session').apply(len)
    sessions = sessions[sessions >= min_answers]
    # sessions = sessions.sample(1)

    for i, mcs in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        for metric_name, metric in mcs.items():
            success_mean = metric.mean()
            s = np.zeros(min_answers)
            for student in sessions.index[:student_count] if student_count else sessions.index:
                current = success_mean
                a = metric.loc[answers[answers['session'] == student].index]
                success = rolling_success(a, success_mean)
                if smooth:
                    success = pd.rolling_mean(pd.Series(success), 20) # hack to make line more smooth
                s += success[:min_answers]
            plt.plot(range(len(s)), s / (student_count if student_count else len(sessions)), label=metric_name)
        plt.legend()


# data = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data = d.Data("../data/matmat/2016-11-28/answers.pd")
data.trim_times()
answers = data.get_dataframe_all()
difficulties = get_difficulties(answers)
time_intensity = get_difficulties(answers,
    model=TimePriorCurrentModel(alpha=0.4, beta=0.04, KC=0.3, KI=0.3, first_level=False),
    data=data, name="time_intensity")

answers = answers.join(difficulties, on='item')
answers = answers.join(time_intensity, on='item')

metrics = [
    {
        "correct": answers['correct'],
        "difficulty": sigmoid(answers['difficulty'] - difficulties.mean()),
    }, {
        "time_intensity": answers['time_intensity'],
        "response time": np.log(answers["response_time"]),
        "time_intensity - response time": answers['time_intensity'] - np.log(answers["response_time"]),
    }, {
        # "difficulty * correct": sigmoid(answers['difficulty'] - difficulties.mean()) * answers['correct'],
        "correct - difficulty": answers['correct'] -sigmoid(answers['difficulty'] - difficulties.mean()),
    }

]

answers_lin = LinearDrop(14).modify(answers.copy())
metrics[-1]["correct, Linear time drop"] = answers_lin['correct']

# answers_bin = TimeLimitResponseModificator([(7, 0.5)]).modify(answers.copy())
# metrics[3]["correct, ternary time"] = answers_bin['correct']


# concept = 'numbers'
concept = 'addition'
# concept = 'multiplication'
plt.title(concept)
answers = answers[answers['item'].isin(items_in_concept(data, concept))]
print(len(answers))

master_curves(answers, metrics, student_count=None, smooth=0)
plt.show()
