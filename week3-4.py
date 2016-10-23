import pandas
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
    f = open('week3-4-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

df = pandas.read_csv('classification.csv')
scores_df = pandas.read_csv('scores.csv')

clfs = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']

y_true = df['true'].values
y_pred = df['pred'].values

# 1

TP = len([1 for [actual, predicted] in df.values if actual == 1 and predicted == 1])
FP = len([1 for [actual, predicted] in df.values if actual == 0 and predicted == 1])
FN = len([1 for [actual, predicted] in df.values if actual == 1 and predicted == 0])
TN = len([1 for [actual, predicted] in df.values if actual == 0 and predicted == 0])

printAndWriteAnswer(1, '{} {} {} {}'.format(TP, FP, FN, TN))

# 2

as_ = accuracy_score(y_true, y_pred)
ps_ = precision_score(y_true, y_pred)
rs_ = recall_score(y_true, y_pred)
fs_ = f1_score(y_true, y_pred)

printAndWriteAnswer(2, '{:.2} {:.2} {:.2} {:.2}'.format(as_, ps_, rs_, fs_))

# 3

def roc_score_for_clf(clf_name):
    return roc_auc_score(scores_df['true'].values, scores_df[clf_name].values)

roc_scores = [(name, roc_score_for_clf(name)) for name in clfs]

best_clf, _ = max(roc_scores, key=lambda t: t[1])

printAndWriteAnswer(3, best_clf)

# 4

def clf_precision(clf_name):
    precision, recall, thresholds = precision_recall_curve(scores_df['true'].values, scores_df[clf_name].values)
    good_recall_indeces = [i for i, value in enumerate(recall) if value > 0.7]
    return max([precision[i] for i in good_recall_indeces])

precisions = [(name, clf_precision(name)) for name in clfs]

best_clf_precision, _ = max(precisions, key=lambda t: t[1])

printAndWriteAnswer(4, best_clf_precision)
