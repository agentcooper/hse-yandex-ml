import pandas
from sklearn import svm

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week3-1-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

df = pandas.read_csv('svm-data.csv', names=['Target', 'Feature1', 'Feature2'])

features = pandas.DataFrame(df, columns=['Feature1', 'Feature2']).values
target = df['Target'].values

clf = svm.SVC(C=100000, random_state=241, kernel='linear')

clf.fit(features, target)

printAndWriteAnswer(1, ' '.join(str(x + 1) for x in clf.support_))
