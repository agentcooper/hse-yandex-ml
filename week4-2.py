import pandas
import numpy as np

import functools

from sklearn import decomposition

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week4-2-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

close_prices_df = pandas.read_csv('close_prices.csv')
djia_index_df = pandas.read_csv('djia_index.csv')

# why?
close_prices_df.drop('date', axis=1, inplace=True)

pca = decomposition.PCA(n_components=10)
pca.fit(close_prices_df.values)

# 1
printAndWriteAnswer(1, next(i + 1 for i, s in enumerate(np.cumsum(pca.explained_variance_ratio_)) if s >= 0.9))

# 2
X_new = pca.transform(close_prices_df.values)
correlation = np.corrcoef(X_new[:,0], djia_index_df['^DJI'].values)

printAndWriteAnswer(2, '{:.2f}'.format(correlation[0][1]))

# 3
company_weight = [(close_prices_df.columns[i], n) for i, n in enumerate(pca.components_[0])]
company_name, _ = max(company_weight, key=lambda t: t[1])

printAndWriteAnswer(3, company_name)
