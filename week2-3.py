import pandas

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week2-3-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

def get_X_y(filename):
    df = pandas.read_csv(filename, names=['Target', 'Feature1', 'Feature2'])
    y = df['Target'].values
    X = pandas.DataFrame(df, columns=['Feature1', 'Feature2']).values
    return (X, y)

def get_score(X_train, X_test):
    clf = Perceptron(random_state = 241)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))

X_train, y_train = get_X_y('perceptron-train.csv')
X_test, y_test = get_X_y('perceptron-test.csv')

score = get_score(X_train, X_test);

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

score_scaled = get_score(X_train_scaled, X_test_scaled)

printAndWriteAnswer(1, round(score_scaled - score, 3))
