from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from models.eloHierarchical import EloHierarchicalModel
from models.time_models import TimeAvgModel, TimeCombiner, TimeStudentAvgModel, TimeItemAvgModel, BasicTimeModel
from utils import data as d
import matplotlib.pylab as plt


def model_learning(prediction_model, time_model, data, length=200):
    enums = defaultdict(lambda: 0)
    model = TimeCombiner(prediction_model, time_model)

    model.pre_process_data(data)
    df = []

    for answer in data.iter():
        item = answer["item"]
        student = answer["student"]
        prediction, time_prediction = model.predict(student, item, answer)
        model.update(student, item, prediction, time_prediction, answer["correct"], answer["response_time"], answer)

        skill = model._prediction_model.get_skill(student)
        speed = model._time_model.get_skill(student)
        enums[student] += 1
        df.append([item, student, skill, speed, enums[student],
                   answer["correct"], np.log(answer["response_time"]),
                   prediction, np.log(time_prediction),
                   model._prediction_model.get_difficulty(item),
                   model._time_model.get_difficulty(item),
                   ])

    df = pd.DataFrame(df, columns=('item', 'student', 'skill', 'speed', 'enum', 'correct', 'response_time_log', 'prediction', 'time_prediction_log', 'difficulty', 'intensity'))

    points = range(1, length + 1)
    plt.subplot(321)
    plt.title(str(model))
    plt.plot(points, [df.loc[df['enum'] == p, 'skill'].mean() for p in points])
    plt.xlabel('# answer')
    plt.ylabel('skill')
    plt.subplot(323)
    plt.plot(points, [df.loc[df['enum'] == p, 'speed'].mean() for p in points])
    plt.xlabel('# answer')
    plt.ylabel('speed')
    plt.subplot(325)
    plt.bar(points[:-1], [(df['enum'] == p).sum() for p in points[:-1]])
    plt.ylabel('User count')
    plt.xlabel('# answer')

    plt.subplot(322)
    plt.plot(points, [df.loc[df['enum'] == p, 'correct'].mean() for p in points], label='observation')
    plt.plot(points, [df.loc[df['enum'] == p, 'prediction'].mean() for p in points], label='prediction')
    plt.legend(loc=3)
    plt.xlabel('# answer')
    plt.ylabel('success')
    plt.subplot(324)
    plt.plot(points, [np.exp(df.loc[df['enum'] == p, 'response_time_log'].mean()) for p in points], label='observation')
    plt.plot(points, [np.exp(df.loc[df['enum'] == p, 'time_prediction_log'].mean()) for p in points], label='prediction')
    plt.legend(loc=3)
    plt.xlabel('# answer')
    plt.ylabel('time')
    plt.subplot(326)
    plt.plot(points, [df.loc[df['enum'] == p, 'intensity'].mean() for p in points])
    plt.ylabel('Intensity')
    # plt.plot(points, [df.loc[df['enum'] == p, 'difficulty'].mean() for p in points])
    # plt.ylabel('Difficulty')
    plt.xlabel('# answer')

    df.to_pickle('model_learning.pd')

    return df


data = d.Data("../data/matmat/2016-06-27/answers.pd")
data.trim_times()
data_long = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data_long.trim_times()



model_learning(
    EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
    BasicTimeModel(alpha=0.6, beta=0.1, K=0.25),
    # TimeItemAvgModel(),
    data
)

plt.show()
