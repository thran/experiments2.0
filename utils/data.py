import inspect
import json
import os
import random
import pandas as pd
import numpy as np
from clint.textui import progress
import sys
import csv
from hashlib import sha1

csv.field_size_limit(sys.maxsize)


class Data():
    def __init__(self, filename, test_subset=False, train_size=None, train_seed=42, only_train=False, only_first=False, filter=None, response_modification=None):
        self._filename = filename
        self._test_subset = test_subset
        self._data = None
        self._data_train = None
        self._data_test = None
        self._train_size = train_size
        self._train_seed = train_seed
        self._only_first = only_first
        self._only_train = only_train
        self._filter = filter
        self._response_modification = response_modification

        self.VERSION = 1

        if not os.path.exists(filename):
            if os.path.exists(filename[:-3] + ".csv"):
                df = pd.read_csv(filename[:-3] + ".csv", engine="python")
                if df["response_time"].mean() > 1000:
                    df["response_time"] /= 1000
                df.to_pickle(filename)
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

    def set_seed(self, seed):
        self._train_seed = 42 + seed

    def set_train_size(self, train_size):
        self._train_size = train_size

    def _load_file(self):
        if self._data is not None:
            return

        self._data = pd.read_pickle(self._filename)
        for req in ["item", "student", "correct", "response_time"]:
            if req not in self._data.columns:
                raise Exception("Column {} missing in {} dataframe".format(req, self._filename))

        if self._only_first:
            self._filter_only_first()

        if self._filter is not None:
            self._filter_data(self._filter[0], self._filter[1])

        if self._response_modification is not None:
            self._data = self._response_modification.modify(self._data)

        if self._test_subset:
            self._data = self._data[:self._test_subset]

        if self._train_size is not None:
            if self._train_seed is not None:
                seed = self._train_seed * (-1 if self._train_seed < 0 else 1)
                random.seed(seed)
                students = self.get_students()
                selected_students = random.sample(students, int(len(students) * self._train_size))
                if self._train_seed < 0:
                    selected_students = list(set(students) - set(selected_students))
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

    def join_predictions(self, df):
        predictions = df[0]
        time_predictions = df[1]
        self._load_file()
        if "prediction" in self._data_test.columns:
            del self._data_test["prediction"]
            del self._data_test["time_prediction"]
        self._data_test = self._data_test.join(pd.Series(predictions, name="prediction"), on="id")
        self._data_test = self._data_test.join(pd.Series(time_predictions, name="time_prediction"), on="id")

        self._data_test['time_prediction_log'] = np.log(self._data_test['time_prediction'])
        self._data_test['response_time_log'] = np.log(self._data_test['response_time'])

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
        if self._data is not None:
            self._data = None
            print("Warning loading data again")
        self._filter = min_answers_per_item, min_answers_per_student
        self._load_file()

    def _filter_data(self, min_answers_per_item=100, min_answers_per_student=10):
        self._load_file()
        self._data = self._data[self._data.join(pd.Series(self._data.groupby("item").apply(len), name="count"), on="item")["count"] > min_answers_per_item]
        self._data = self._data[self._data.join(pd.Series(self._data.groupby("student").apply(len), name="count"), on="student")["count"] > min_answers_per_student]

    def only_first(self):
        if self._data is not None:
            self._data = None
            print("Warning loading data again")
        self._only_first = True
        self._load_file()

    def _filter_only_first(self):
        filtered = self._data.drop_duplicates(['student', 'item'])
        filtered.loc[:,"index"] = filtered["id"]
        filtered.set_index("index", inplace=True)
        self._data = filtered

    def trim_times(self, limit=60):
        self._load_file()
        self._data.loc[self._data["response_time"] < 0.5, "response_time"] = 0.5
        self._data.loc[self._data["response_time"] > limit, "response_time"] = limit

    def add_log_response_times(self):
        self._load_file()
        self._data["log_response_time"] = np.log(self._data["response_time"])

    def get_skill_structure(self, filename="skills.csv"):
        skills = self.get_skills_df(filename)
        map = {}
        for id, skill in skills.iterrows():
            map[id] = int(skill["parent"]) if not pd.isnull(skill["parent"]) else None
        return map

    def get_skills_df(self, filename="skills.csv"):
        file = self._filename.split("/")
        file[-1] = filename
        return pd.read_csv(os.path.join(*file), index_col="id")

    def get_skill_id(self, skill):
        df = self.get_skills_df()
        pk = df[df["identifier"] == skill].index[0]
        level = 0
        current = pk
        while not np.isnan(df.loc[current]["parent"]):
            level += 1
            current = int(df.loc[current]["parent"])
        return pk, level

    def get_item_assignment(self, filename="items.csv"):
        items = self.get_items_df(filename)
        return dict(zip(items.index, items["skill"]))

    def get_items_df(self, filename="items.csv", with_skills=True):
        file = self._filename.split("/")
        file[-1] = filename
        items = pd.read_csv(os.path.join(*file), index_col=0)
        if not with_skills:
            return items
        skills = self.get_skills_df()
        return items.join(skills, on="skill")


    def get_concepts(self, level=1):
        items = self.get_items_df()
        skills = self.get_skills_df()

        concepts = {}
        for concept in items["skill_lvl_" + str(level)].unique():
            concepts[skills.loc[concept, "name"]] = list(items[items["skill_lvl_" + str(level)] == concept].index)
        return concepts


class ResponseModificator():
    def __init__(self):
        self._name = "ResMod"

    def modify(self, data):
        return data

    def __str__(self):
        s = self._name
        (args, _, _, defaults) = inspect.getargspec(self.__init__)
        defaults = defaults if defaults else []
        s += "".join([", {}:{}".format(a, getattr(self, "_" + a)) for a, d in zip(args[-len(defaults):], defaults) if getattr(self, "_" + a) != d])
        return s


class BinaryResponse(ResponseModificator):
    def __init__(self, threshold=0):
        super().__init__()
        self._name = "Binary"
        self._threshold = threshold

    def modify(self, data):
        data.loc[data["correct"] > self._threshold, "correct"] = 1
        return data


class TimeLimitResponseModificator(ResponseModificator):
    def __init__(self, limits=None):
        super().__init__()
        self._limits = limits
        self._name = "Discrete"

    def modify(self, data):
        for limit, value in self._limits:
            data.loc[(data["response_time"] > limit) & (data["correct"] > 0), "correct"] = value
        return data


class ExpDrop(ResponseModificator):
    def __init__(self, expected_time=None, slope=None):
        super().__init__()
        self._expected_time = expected_time
        self._slope = slope
        self._name = "Exp"

    def modify(self, data):
        data.loc[(data["response_time"] > self._expected_time) & (data["correct"] > 0), "correct"] = \
            self._slope ** ((data["response_time"] / self._expected_time) - 1)
        return data


class LinearDrop(ResponseModificator):
    def __init__(self, max=None):
        super().__init__()
        self._max = max
        self._name = "linear"

    def modify(self, data):
        data["correct2"] = data["correct"]
        data.loc[(data["response_time"] > self._max) & (data["correct"] > 0), "correct"] = 0
        data.loc[data["correct"] > 0, "correct"] = (self._max - data["response_time"]) / float(self._max)
        return data


class MathGardenResponseModificator(ResponseModificator):
    def __init__(self, max=None):
        super().__init__()
        self._max = max
        self._name = "MathGarden"

    def modify(self, data):
        data["correct2"] = data["correct"]
        data.loc[data["correct2"] == 1, "correct"] = (self._max - data["response_time"]) / float(self._max)
        data.loc[data["correct2"] == 0, "correct"] = -(self._max - data["response_time"]) / float(self._max)
        data.loc[(data["response_time"] > self._max), "correct"] = 0
        return data


def filter_students_with_many_answers(number_of_answers=50):
    return lambda data: data[data.join(pd.Series(data.groupby("student").apply(len), name="count"), on="student")["count"] >= number_of_answers]


def transform_response_by_time(limits=None, binarize_before=False):
    def fce(data):
        data = data.copy(True)
        m = TimeLimitResponseModificator(limits)
        if binarize_before:
            data = response_as_binary()(data)
        return m.modify(data)

    return fce


def transform_response_by_time_linear(max=None):
    def fce(data):
        data = data.copy(True)
        m = LinearDrop(max=max)
        return m.modify(data)

    return fce

def response_as_binary():
    def fce(data):
        data = data.copy(True)
        m = BinaryResponse()
        return m.modify(data)

    return fce


def compute_corr(data, min_periods=1, method="pearson", merge_skills=False):
    locs = locals()
    name = "; ".join(["{}:{}".format(name, str(locs[name])) for name in sorted(locs.keys())])
    hash = sha1(name.encode()).hexdigest()[:20]
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "{}.corr.pd".format(hash))
    print(filename, name)
    if os.path.exists(filename):
        return pd.read_pickle(filename)

    df = data.get_dataframe_train()
    if not merge_skills:
        corr = df.pivot("student", "item", "correct").corr(method=method, min_periods=min_periods)
        corr.to_pickle(filename)
        return corr
    else:
        items = data.get_items_df()
        df = df.join(items["skill_lvl_3"], on="item")
        df = df[~df["skill_lvl_3"].isnull()]
        df = pd.DataFrame(df.groupby(["student", "skill_lvl_3"])["correct"].mean())
        corr = df.reset_index().pivot("student", "skill_lvl_3", "correct").corr(method=method, min_periods=min_periods)
        corr.to_pickle(filename)
        return corr


def convert_slepemapy(filename):
    answers = pd.read_csv(filename)
    answers["correct"] = answers["item_asked_id"] == answers["item_answered_id"]
    answers.rename(columns={
        "time": "timestamp",
        "item_asked_id": "item",
        "user_id": "student",
        "item_answered_id": "answer",
    }, inplace=True)
    answers["response_time"] /= 1000
    answers.to_pickle(filename.replace("csv", "pd"))
    return answers

def convert_prosoapp(filename):
    answers = pd.read_csv(filename)
    answers["correct"] = answers["item_asked"] == answers["item_answered"]
    answers.rename(columns={
        "time": "timestamp",
        "user": "student",
        "item_answered": "answer",
    }, inplace=True)
    answers["response_time"] /= 1000
    answers.to_pickle(filename.replace("csv", "pd"))
    return answers

def items_in_concept(data, concept):
    pk, level = data.get_skill_id(concept)
    items = data.get_items_df()
    return items[items["skill_lvl_" + str(level)] == pk].index