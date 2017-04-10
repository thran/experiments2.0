import pandas as pd

answers = pd.read_csv('2016-ab-target-difficulty/answers.csv')
items = pd.read_csv('2016-ab-target-difficulty/flashcards.csv')

items = items[(items['context_name'] == 'Europe') & (items['term_type'] == 'state')]
answers = answers[answers['item_id'].isin(items['item_id'])]
answers['correct'] = (answers['item_id'] == answers['item_answered_id']) * 1
answers = answers.loc[:, ['item_id', 'user_id', 'guess', 'correct']]

print(answers)

items.to_pickle('items.pd')
answers.to_pickle('answers.pd')

print(len(answers))
