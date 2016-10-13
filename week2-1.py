import pandas;
import numpy;

from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;

from sklearn.preprocessing import scale;

from sklearn.neighbors import KNeighborsClassifier;

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week2-1-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

features = [
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
];

df = pandas.read_csv('wine.data', names=['ClassId'] + features);

features = pandas.DataFrame(df, columns=features);

X = features.values;
y = df['ClassId'].values;

def get_score(X, k):
    knn = KNeighborsClassifier(n_neighbors=k);
    kf = KFold(n_splits=5, shuffle=True, random_state=42);
    scores = cross_val_score(knn, X, y, cv=kf);
    return numpy.mean(scores);

def get_best_score(X):
    scores = [(k, get_score(X, k)) for k in range(1, 50 + 1)];
    scores_sorted = sorted(scores, key=lambda tuple: tuple[1]);
    k, top_score = scores_sorted[-1];
    return (k, top_score);

k_normal, score_normal = get_best_score(X);

k_scaled, score_scaled = get_best_score(scale(X));

printAndWriteAnswer(1, k_normal);
printAndWriteAnswer(2, round(score_normal, 2));

printAndWriteAnswer(3, k_scaled);
printAndWriteAnswer(4, round(score_scaled, 2));
