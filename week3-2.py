import numpy as np

from sklearn import datasets, feature_extraction, svm

from sklearn.model_selection import KFold, GridSearchCV;

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week3-2-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = feature_extraction.text.TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(n_splits=5, shuffle=True, random_state=241)

clf = svm.SVC(kernel='linear', random_state=241)

gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

print('Fitting, might take a while...')
gs.fit(X, y)

coefs = gs.best_estimator_.coef_.toarray()[0]

absolute_scores = [(index, abs(score)) for index, score in enumerate(coefs)]

sorted_scores = sorted(absolute_scores, key=lambda tuple: tuple[1])

feature_mapping = vectorizer.get_feature_names()

top_words = [feature_mapping[index] for index, score in sorted_scores[-10:]]

printAndWriteAnswer(1, ' '.join(sorted(top_words)))
