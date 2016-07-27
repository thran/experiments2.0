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
        df.append([item, student, skill, speed, enums[student]])

    df = pd.DataFrame(df, columns=('item', 'student', 'skill', 'speed', 'enum'))

    points = range(1, length + 1)
    plt.subplot(311)
    plt.plot(points, [df.loc[df['enum'] == p, 'skill'].mean() for p in points])
    plt.xlabel('# answer')
    plt.ylabel('skill')
    plt.subplot(312)
    plt.plot(points, [df.loc[df['enum'] == p, 'speed'].mean() for p in points])
    plt.xlabel('# answer')
    plt.ylabel('speed')
    plt.subplot(313)
    plt.bar(points[:-1], [(df['enum'] == p).sum() for p in points[:-1]])
    plt.ylabel('User count')
    plt.xlabel('# answer')

    return df


data = d.Data("../data/matmat/2016-06-27/answers.pd")
data.trim_times()
data_long = d.Data("../data/matmat/2016-06-27/answers.pd", filter=(100, 100))
data_long.trim_times()



model_learning(
    EloHierarchicalModel(KC=1, KI=0.75, alpha=0.8, beta=0.02),
    BasicTimeModel(alpha=0.6, beta=0.1, K=0.25),
    data
)

plt.show()
