import sys

import pandas
import numpy
import scipy

from sklearn.feature_extraction import text, DictVectorizer
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week4-1-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

try:
    salary_train_df = pandas.read_csv('salary-train.csv')
except:
    print('"salary-train.csv" can be found at:\nhttps://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/QFvJY/linieinaia-rieghriessiia-proghnoz-oklada-po-opisaniiu-vakansii')
    sys.exit(1)

salary_test_df = pandas.read_csv('salary-test-mini.csv')

salary_train_df['LocationNormalized'].fillna('nan', inplace=True)
salary_train_df['ContractTime'].fillna('nan', inplace=True)

def prepare_text(dataframe):
    return [
        text.lower() for text in dataframe['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
    ]

def prepare_dict(dataframe):
    return dataframe[['LocationNormalized', 'ContractTime']].to_dict('records')

dv = DictVectorizer()
tfidf = text.TfidfVectorizer(min_df=5)

LocationNormalized_ContractTime_train = dv.fit_transform(prepare_dict(salary_train_df))
LocationNormalized_ContractTime_test = dv.transform(prepare_dict(salary_test_df))

FullDescription_tfidf_train = tfidf.fit_transform(prepare_text(salary_train_df))
FullDescription_tfidf_test = tfidf.transform(prepare_text(salary_test_df))

X_train = scipy.sparse.hstack([
    FullDescription_tfidf_train,
    LocationNormalized_ContractTime_train
])

X_test = scipy.sparse.hstack([
    FullDescription_tfidf_test,
    LocationNormalized_ContractTime_test
])

y_train = salary_train_df['SalaryNormalized'].values

clf = Ridge(alpha=1, random_state=241)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

printAndWriteAnswer(1, ' '.join(['{:.2f}'.format(n) for n in prediction]))
