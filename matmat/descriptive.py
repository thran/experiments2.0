from utils.data import Data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = Data("../data/matmat/2015-11-20/answers.pd")


def response_times(data):
    data.filter_data()
    data.trim_times()
    data.add_log_response_times()
    df = data.get_dataframe_all()

    plt.figure()
    sns.distplot(df["response_time"], hist=False, bins=30, label="all")
    sns.distplot(df[df["answer"].isnull()]["response_time"], hist=False, bins=30, label="without answer (null)")
    sns.distplot(df[df["correct"] == True]["response_time"], hist=False, bins=30, label="correct answer")
    sns.distplot(df[df["correct"] == False][~df["answer"].isnull()]["response_time"], hist=False, bins=30, label="wrong answer (not null)")
    plt.title("Response time distribution per answer type")

    plt.figure()
    sns.distplot(np.exp(df.groupby("item")["log_response_time"].mean()), hist=False, label="items ({})".format(len(data.get_items())))
    sns.distplot(np.exp(df.groupby("student")["log_response_time"].mean()), hist=False, label="students ({})".format(len(data.get_students())))
    sns.distplot(np.exp(df[~df["answer"].isnull()].groupby("item")["log_response_time"].mean()), hist=False, label="items - with answer({})".format(len(data.get_items())))
    sns.distplot(np.exp(df[~df["answer"].isnull()].groupby("student")["log_response_time"].mean()), hist=False, label="students - with answer ({})".format(len(data.get_students())))
    plt.title("Distribution of median time (exp of median of log times)")


def answer_count(data, per_student=True, per_item=True, student_drop_off=True):
    # data.filter_data()
    df = data.get_dataframe_all()

    if per_student:
        plt.figure()
        sns.distplot(df.groupby("student").size(), kde=False, bins=30, label="", hist_kws={"range": [0, 300]})
        plt.xlabel("answer count")
        plt.ylabel("student count")
        plt.title("Answer count distribution per student")

    if per_item:
        plt.figure()
        sns.distplot(df.groupby("item").size(), kde=False, bins=300, label="", hist_kws={"range": [0, 300]})
        plt.xlabel("answer count")
        plt.ylabel("item count")
        plt.title("Answer count distribution per item")

    if student_drop_off:
        plt.figure()
        counts = df.groupby("student").size()
        r = range(1, 100)
        plt.plot(r, [sum(counts.values >= count) / len(counts) for count in r])
        plt.xlabel("answer count")
        plt.ylabel("percentage of students")
        plt.title("Student drop-off")

# response_times(data)
answer_count(data, per_student=False, per_item=False)

plt.show()