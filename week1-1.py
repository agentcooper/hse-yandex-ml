import pandas;

import numpy as np;

import re;

from collections import Counter;

from pprint import pprint;

data = pandas.read_csv('titanic.csv', index_col='PassengerId');

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"');
    f = open('week1-1-' + str(index) + '.txt', 'w');
    f.write(str(answer));
    f.close();

#1
nr_females = len(data[data.Sex == 'female']);
nr_males = len(data[data.Sex == 'male']);
printAndWriteAnswer(1, '{} {}'.format(nr_males, nr_females));

#2
survived_percentage = len(data[data.Survived == 1]) / len(data) * 100;
printAndWriteAnswer(2, round(survived_percentage, 2));

#3
first_class_percentage = len(data[data.Pclass == 1]) / len(data) * 100;
printAndWriteAnswer(3, round(first_class_percentage, 2));

#4
age_mean = data['Age'].mean();
age_median = data['Age'].median();
printAndWriteAnswer(4, '{} {}'.format(round(age_mean, 2), round(age_median, 2)));

#5
SibSp_Parch_correlation = data.corr(method='pearson')['SibSp']['Parch'];
printAndWriteAnswer(5, round(SibSp_Parch_correlation, 2));

#6
def getFirstName(fullName):
    # Serepeca, Miss. Augusta
    m = re.search('Miss\.\s?(\w+)', fullName);
    if m:
        return m.group(1);

    # Arnold-Franchi, Mrs. Josef (Josefine Franchi)
    # Boulos, Mrs. Joseph (Sultana)
    m = re.search('\((\w+).*\)', fullName);
    if m:
        return m.group(1);

    # Masselmani, Mrs. Fatima
    m = re.search('^\w+,\s(Mrs|Ms)\. (\w+)$', fullName);
    if m:
        return m.group(2);

    # Moor, Mrs. (Beila)
    m = re.search('(Mrs|Ms). \((\w+).*\)$', fullName);
    if m:
        return m.group(2);

    return '?';

women_names = data[data.Sex == 'female']['Name'];
first_names = list(map(getFirstName, women_names.values));

full_names_to_first = dict(
    zip(women_names.values, first_names)
);

# unknown_first_names = { k: v for k, v in full_names_to_first.items() if v == '?' };
# pprint(unknown_first_names);

[(top_first_name, nr_occurance)] = Counter(first_names).most_common(1);

printAndWriteAnswer(6, top_first_name);
