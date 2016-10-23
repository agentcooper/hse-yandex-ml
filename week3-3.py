import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
import math

df = pandas.read_csv('data-logistic.csv', names=['Target', 'X1', 'X2'])

X = np.array(pandas.DataFrame(df, columns=['X1', 'X2']).values)
y = np.array(df['Target'].values)

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week3-3-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

def gradient_descent(X, y, eps, k, C, initial = [0, 0], max_iterations = 1000):

    def compute_w(j, w):
        l = len(y)

        # regularization
        r = k * C * w[j];

        S = [
            y[i] *
            X[i][j] *
            (1 - 1 / (1 + np.exp(-y[i]*X[i].dot(w))))

            for i in range(0, l)
        ]

        return w[j] + k*(1/l) * np.sum(S) - r;

    w = np.array(initial);

    for i in range(1, max_iterations):
        next_w = np.array([compute_w(0, w), compute_w(1, w)])

        if (np.linalg.norm(next_w - w) < eps):
            return next_w
        else:
            w = next_w

    return w

def sigmoid(x, w):
     return 1 / (1 + np.exp(-x.dot(w)))

def score_with_regularization(C):
    w = gradient_descent(X, y, 1e-5, 0.1, C);

    y_scores = [sigmoid(x, w) for x in X];

    return roc_auc_score(y, y_scores)

printAndWriteAnswer(1,
    '{} {}'.format(round(score_with_regularization(0), 3), round(score_with_regularization(10), 3))
    )
