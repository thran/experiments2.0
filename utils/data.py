import inspect
import json
import os
import random
import pandas as pd
import numpy as np
from clint.textui import progress
import sys
import csv

csv.field_size_limit(sys.maxsize)


class Data():
    def __init__(self, filename, test_subset=False, train_size=None, train_seed=42, only_train=False):
        self._filename = filename
        self._test_subset = test_subset
        self._data = None
        self._data_train = None
        self._data_test = None
        self._train_size = train_size
        self._train_seed = train_seed
        self._only_train = only_train

        self.VERSION = 1

        if not os.path.exists(filename):
            if os.path.exists(filename[:-3] + ".csv"):
                pd.read_csv(filename[:-3] + ".csv", engine="python").to_pickle(filename)
            else:
                raise FileNotFoundError("Data file '{}' not found".format(filename))

    def __str__(self):
        s = "Data.{}: {}".format(self.VERSION, self._filename)
        (args, _, _, defaults) = inspect.getargspec(self.__init__)
        s += "".join([", {}:{}".format(a, getattr(self, "_" + a)) for a, d in zip(args[-len(defaults):], defaults) if getattr(self, "_" + a) != d])
        return s

    def get_dataframe_all(self):
        self._load_file()
        return self._data

    def get_dataframe_train(self):
        self._load_file()
        return self._data_train

    def get_dataframe_test(self):
        self._load_file()
        return self._data_test

    def _load_file(self):
        if self._data is not None:
            return

        self._data = pd.read_pickle(self._filename)
        for req in ["item", "student", "correct", "response_time"]:
            if req not in self._data.columns:
                raise Exception("Column {} missing in {} dataframe".format(req, self._filename))

        if self._test_subset:
            self._data = self._data[:self._test_subset]

        if self._train_size is not None:
            if self._train_seed:
                random.seed(self._train_seed)
                students = self.get_students()
                selected_students = random.sample(students, int(len(students) * self._train_size))
            else:
                selected_students = json.load(open(os.path.join(os.path.dirname(self._filename), "/train_students.json")))
            self._data_train = self._data[self._data["student"].isin(selected_students)]
            self._data_test = self._data[~self._data["student"].isin(selected_students)]
        else:
            self._data_test  = self._data
            self._data_train = pd.DataFrame(columns=self._data.columns)

        if self._only_train:
            self._data = self._data_train
            self._data_test = pd.DataFrame(columns=self._data.columns)

    def join_predictions(self, predictions):
        self._load_file()
        if "prediction" in self._data_test.columns:
            del self._data_test["prediction"]
        self._data_test = self._data_test.join(pd.Series(predictions, name="prediction"), on="id")

    def get_items(self):
        self._load_file()
        return list(self._data["item"].unique())

    def get_students(self):
        self._load_file()
        return list(self._data["student"].unique())

    def iter(self, data=None):
        self._load_file()
        if data is None:
            data = self._data
        columns = data.columns.values
        for row in progress.bar(data.values, every=1000):
            yield dict(zip(columns, row))

    def iter_train(self):
        self._load_file()
        return self.iter(self._data_train)

    def iter_test(self):
        self._load_file()
        return self.iter(self._data_test)

    def train_size(self):
        return len(self.get_dataframe_train())

    def test_size(self):
        return len(self.get_dataframe_test())

    def size(self):
        return len(self.get_dataframe_all())

    def filter_data(self, min_answers_per_item=100, min_answers_per_student=10):
        self._load_file()
        self._data = self._data[self._data.join(pd.Series(self._data.groupby("student").apply(len), name="count"), on="student")["count"] > min_answers_per_student]
        self._data = self._data[self._data.join(pd.Series(self._data.groupby("item").apply(len), name="count"), on="item")["count"] > min_answers_per_item]


    def trim_times(self, limit=60):
        self._load_file()
        self._data.loc[self._data["response_time"] < 0.5, "response_time"] = 0.5
        self._data.loc[self._data["response_time"] > limit, "response_time"] = limit

    def add_log_response_times(self):
        self._load_file()
        self._data["log_response_time"] = np.log(self._data["response_time"])

    def get_skill_structure(self, filename="skills.csv"):
        file = self._filename.split("/")
        file[-1] = filename
        skills = pd.read_csv(os.path.join(*file), index_col="id")
        map = {}
        for id, skill in skills.iterrows():
            map[id] = int(skill["parent"]) if not pd.isnull(skill["parent"]) else None
        return map

    def get_item_assignment(self, filename="items.csv"):
        file = self._filename.split("/")
        file[-1] = filename
        items = pd.read_csv(os.path.join(*file), index_col="id")
        return dict(zip(items.index, items["skill"]))


def filter_students_with_many_answers(number_of_answers=50):
    return lambda data: data[data.join(pd.Series(data.groupby("student").apply(len), name="count"), on="student")["count"] >= number_of_answers]