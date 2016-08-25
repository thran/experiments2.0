import pandas as pd

dir_name = '2016-05-18'

answers = pd.read_csv(dir_name + '/doplnovackaLog.csv', sep=';')
answers = answers.groupby(['user', 'word']).first().reset_index()
questions = pd.read_csv(dir_name + '/doplnovackaZadani.csv', sep=';', index_col='word')
questions['question'] = questions.index
questions = questions.set_index('id')

question_solutions = {}
for question in questions.itertuples():
    question_solutions[question[0]] = question[3] if question[6].replace('_', question[3]) == question[2] else question[4]
    questions.loc[question[0], 'correct'] = question_solutions[question[0]]

questions.loc[questions['variant1'] == questions['correct'], 'correct_variant'] = 0
questions.loc[questions['variant2'] == questions['correct'], 'correct_variant'] = 1

del answers['microtime']
del questions['Unnamed: 6']
questions['correct_variant'] = questions['correct_variant'].astype(int)
questions = questions.rename(columns={'correct': 'solution'})

print(questions.columns)
questions.to_pickle(dir_name + '/items.pd')

questions['id'] = questions.index
questions = questions.set_index('question')
answers = answers.join(questions, on='word', rsuffix='_item')

answers = answers.loc[:, ['id', 'user', 'correct']].rename(columns={'id': 'item', 'user': 'student'})
print(answers.columns)
answers.to_pickle(dir_name + '/answers.pd')