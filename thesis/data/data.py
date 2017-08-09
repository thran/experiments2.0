from utils import data

# data.convert_slepemapy('slepemapy/answers.csv')
# data.convert_prosoapp('anatom/answers.csv')

matmat = data.Data('matmat/answers.pd', filter=(100, 10))
slepemapy = data.Data('slepemapy/answers.pd', filter=(100, 10))
anatom = data.Data('anatom/answers.pd', filter=(100, 10))
umimecesky = data.Data('umimecesky/answers.pd', filter=(100, 10))

mathgarden_addition = data.Data('mathgarden/addition.pd', filter=(100, 10))
mathgarden_multiplication = data.Data('mathgarden/multiplication.pd', filter=(100, 10))
mathgarden_subtraction = data.Data('mathgarden/subtraction.pd', filter=(100, 10))

# print(matmat.get_dataframe_all().columns)
# print(slepemapy.get_dataframe_all().columns)
# print(anatom.get_dataframe_all().columns)
#


def basic_stats(dataset):
    df = dataset.get_dataframe_all()
    print(df.columns)
    print('learners:', len(df['student'].unique()))
    print('items:', len(df['item'].unique()))
    print('answers:', len(df))
    print('SR:', df['correct'].mean())
    print('')
    print('')

    # print('\\numprint{{{}}} &  \\numprint{{{}}} & \\numprint{{{}}} \\\\'.format(len(df['student'].unique()), len(df['item'].unique()), len(df)))

print('slepemapy')
basic_stats(slepemapy)
print('matmat')
basic_stats(matmat)
print('anatom')
basic_stats(anatom)
print('umimecesky')
basic_stats(umimecesky)

print('mathgarden_addition')
basic_stats(mathgarden_addition)

print('mathgarden_multiplication')
basic_stats(mathgarden_multiplication)

print('mathgarden_subtraction')
basic_stats(mathgarden_subtraction)
