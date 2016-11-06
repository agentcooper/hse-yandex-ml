import pandas
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week5-2-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

df = pandas.read_csv('gbm-data.csv')

X = np.array(pandas.DataFrame(df, columns=df.columns[1:]).values)
y = np.array(df['Activity'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(y_pred):
     return 1 / (1 + np.exp(-y_pred))

rates = [1, 0.5, 0.3, 0.2, 0.1]

r = 0.2

clf = GradientBoostingClassifier(n_estimators=250, learning_rate=r, verbose=True, random_state=241)
clf.fit(X_train, y_train)

test_loss = [
    (i, log_loss(y_test, sigmoid(y_pred))) for i, y_pred in enumerate(clf.staged_decision_function(X_test))
]

train_loss = [
    (i, log_loss(y_train, sigmoid(y_pred))) for i, y_pred in enumerate(clf.staged_decision_function(X_train))
]

#
# plt.figure()
# plt.plot([loss for i, loss in test_loss], 'r', linewidth=2)
# plt.plot([loss for i, loss in train_loss], 'g', linewidth=2)
# plt.legend(['test', 'train'])
# plt.show();

printAndWriteAnswer(1, 'overfitting')

min_n, min_loss = min(test_loss, key=lambda t: t[1])

printAndWriteAnswer(2, '{:.2} {}'.format(min_loss, min_n))

rf_clf = RandomForestClassifier(n_estimators=min_n, random_state=241)
rf_clf.fit(X_train, y_train)

rf_loss = log_loss(y_test, rf_clf.predict_proba(X_test))

printAndWriteAnswer(3, '{:.2}'.format(rf_loss))
