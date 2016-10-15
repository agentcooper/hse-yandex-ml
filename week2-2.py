import numpy;

from sklearn import datasets, preprocessing, neighbors;

from sklearn.model_selection import KFold, cross_val_score;

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week2-2-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

dataset = datasets.load_boston();

data_scaled = preprocessing.scale(dataset.data);

X = data_scaled;
y = dataset.target;

def get_score(p):
    knr = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=p);
    kf = KFold(n_splits=5, shuffle=True, random_state=42);
    scores = cross_val_score(knr, X, y, cv=kf, scoring='neg_mean_squared_error');
    return numpy.mean(scores);

scores = [(p, get_score(p)) for p in numpy.linspace(1, 10, num=200)];

scores_sorted = sorted(scores, key=lambda tuple: tuple[1]);

best_p, best_score = scores_sorted[-1];

printAndWriteAnswer(1, round(best_p, 2));
