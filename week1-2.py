import pandas;
import numpy as np;
from sklearn.tree import DecisionTreeClassifier;

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week1-2-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

df = pandas.read_csv('titanic.csv', index_col='PassengerId');

sex_to_number = {
    'male': 1,
    'female': 2,
};

df_columns = pandas.DataFrame(df, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived']).dropna();
df_columns['Sex'] = [sex_to_number[sex] for sex in df_columns.Sex];

df_features = pandas.DataFrame(df_columns, columns=['Pclass', 'Fare', 'Age', 'Sex']);
df_target = pandas.DataFrame(df_columns, columns=['Survived']);

X = np.array(df_features.values);
y = np.array(df_target.values);

clf = DecisionTreeClassifier(random_state = 241);

clf.fit(X, y);

features_by_importance = [{'importance': i, 'name': n} for i, n in zip(clf.feature_importances_, df_features.columns.values)];

sorted_features = sorted(features_by_importance, key=lambda feature: feature['importance']);

sorted_feature_names = [feature['name'] for feature in sorted_features];

printAndWriteAnswer(1, ' '.join(sorted_feature_names[-2:]));
