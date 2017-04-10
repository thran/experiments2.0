from utils.utils import sigmoid


def success_rate(answers):
    curve = []
    count = 0
    corrects = 0
    for correct in answers['correct']:
        if correct:
            corrects += 1
        count += 1
        curve.append(corrects / count)
    return curve


def exponential_average(answers, initial_value=0, exp=0.9, column='correct'):
    current = initial_value
    results = []
    i = 0
    for value in answers[column]:
        current = current * exp + (1 - exp) * value
        i += 1
        results.append(current)
    return results


exponential_average({'correct': [1] * 100})

def exponential_average_time(answers, **kwargs):
    return exponential_average(answers, column='correct_time', **kwargs)


def exponential_average_difficulty(answers, **kwargs):
    column = 'correct_difficulty'
    answers[column] = answers['correct'] - (1 - sigmoid(answers['difficulty']))
    return exponential_average(answers, column=column, **kwargs)


def exponential_average_difficulty_time(answers, **kwargs):
    column = 'correct_difficulty_time'
    answers[column] = answers['correct_time'] - (1 - sigmoid(answers['difficulty_time']))
    return exponential_average(answers, column=column, **kwargs)


# stupid idea
def model_prediction(answers, column='prediction'):
    return list(answers[column])[1:]


def model_skills(answers, avg_difficulty=0, **kwargs):
    return list(sigmoid(answers['skill'] - avg_difficulty))


def model_skills_time(answers, avg_difficulty=0, **kwargs):
    return list(sigmoid(answers['skill_time'] - avg_difficulty))
