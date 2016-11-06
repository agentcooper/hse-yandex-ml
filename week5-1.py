import pandas
import numpy

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week5-1-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

df = pandas.read_csv('abalone.csv')

df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

features = pandas.DataFrame(df, columns=df.columns[:-1])
target = df['Rings']

def get_score(n_estimators):
    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=1);
    scores = cross_val_score(clf, features.values, target.values, cv=kf, scoring='r2');
    return numpy.mean(scores);

min_n = next(n for n in range(1, 50 + 1) if get_score(n) >= 0.52)

printAndWriteAnswer(1, min_n)
