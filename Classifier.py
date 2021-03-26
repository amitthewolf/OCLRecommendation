## Importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sqlite3
import pandas as pd
import numpy as np
from time import time
from scipy.stats import entropy
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from dataExtractor import dataExtractor as DataExtractor
from DAO import DAO
from TestConfig import TestConfig
from node2vec import node2vec
from sklearn.model_selection import cross_val_score
from itertools import chain, combinations

def featureImportance(feature_names):
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    model = clf.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=20, fontsize=8)
    plt.title("Feature Importance")
    plt.show()


def classify(X_train, X_test, y_train, y_test):
    models = [ GaussianNB(),KNeighborsClassifier(),RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced') ]
    for model in models:
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        print(model.__class__.__name__ + " : ")
        print(" Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
        print(" Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))
        print( '-' * 50)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def run(test_config):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print("-" * 50)
    print("Chosen features :    " + str(list(df.columns)))
    print("-" * 50)
    print("Number of positive target records : " + str(df[df['ContainsConstraints'] == 1].shape[0]))
    print("Number of negative target records : " + str(df[df['ContainsConstraints'] == 0].shape[0]))

    # for equal for target variable
    if test_config.sampling_strategy == 'under':
        # Under-sample the majority
        sampler = RandomUnderSampler()
    else:
        # over-sample the minority
        sampler = RandomOverSampler()

    X, y = sampler.fit_resample(X, y)

    print("-" * 25 + "Mutual Information" + "-" * 25)
    feature_names = df.columns
    res = mutual_info_classif(X, y)
    print(dict(zip(feature_names, res)))

    # #Split to Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)

    print("-" * 25 + " Sampling Data " + "-" * 25)
    print("Number of Rows in Data after sampling is: " + str(X.shape[0]))

    print("-" * 25 + " Data stats " + "-" * 25)
    print("Number of Rows in Train Set is : " + str(X_train.shape[0]))
    print("Number of Rows in Test Set is : " + str(X_test.shape[0]))
    print()

    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X_train, y_train, cv=test_config.cross_val_k)
    print("Cross-Validation k = {} : ".format(test_config.cross_val_k))
    print('Scores :  {} '.format(scores))
    print("%0.2f average accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # sys.stdout = open('outputs/outputs.txt', 'a')

    print(datetime.now())
    # str1 = ''.join(featuresNames)
    # print("features: "+ ''.join(featuresNames))
    print("features: " + str(list(df.columns)))

    # n2v_feat = "Features: " + n2v_section['n2v_features_num'] + ", Attributes: " + n2v_section['n2v_use_attributes'] + \
    #            ", Inheritance: " + n2v_section['n2v_use_inheritance'] + ", Return weight: " + \
    #            n2v_section['n2v_return_weight'] + ", Walklen: " + n2v_section['n2v_walklen'] + ", Epcochs:" + \
    #            n2v_section['n2v_epochs'] + ", Neighbour weight: " + n2v_section['n2v_neighbor_weight'] + \
    #            ", Use PCA: " + n2v_section['use_pca'] + ", PCA num: " + n2v_section['pca_num']
    #
    # if n2v_section['n2v_flag'] == 'True':
    #     print('Node2Vec Features:')
    #     print(n2v_feat)

    print("sampling strategy: " + test_config.sampling_strategy)
    print("-" * 25 + " Results " + "-" * 25)
    classify(X_train, X_test, y_train, y_test)
    # sys.stdout.close()

dao = DAO()
config = ConfigParser()
dataExtractor = DataExtractor()

#get configurations
config.read('conf.ini')
fixed_section = config['fixed_params']
iterations = int(fixed_section['iterations'])
random_param_sampling = fixed_section['random']

if random_param_sampling == 'True':
    test_config = TestConfig()
else:
    test_config = TestConfig(random=False)

featureNames = test_config.classifier_section['featureNames'].split(',')
target = test_config.classifier_section['Target'].split(',')

df = dao.getObjects()

for i in range(iterations):
    print('*' * 50)
    print("{} Experiment ".format(i + 1))
    print('*' * 50)
    test_config.update_iteration_params(i)
    df = dataExtractor.Set_N2V_DF(df, test_config)
    df = dataExtractor.get_final_df(df,featureNames, target)
    run(test_config)
    print("classifier params")
    print(test_config.sampling_strategy)
    print(test_config.test_ratio)
    print(test_config.cross_val_k)
    print("n2v params")
    print(test_config.n2v_features_num)
    print(test_config.n2v_return_weight)
    print(test_config.n2v_walklen)
    print(test_config.n2v_epochs)
    print(test_config.n2v_neighbor_weight)
    print(test_config.pca)