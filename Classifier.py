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

def feature_imp(Final):
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(X,y)
    for name, importance in zip(Final.columns, rnd_clf.feature_importances_):
        ...
        print(name, "=", importance)
    features = Final.columns
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
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


dao = DAO()
config = ConfigParser()
dataExtractor = DataExtractor()
config.read('conf.ini')
conf = config['classifier']
sampling_strategy = conf['sampling']
sys.stdout = open('outputs/outputs.txt', 'a')
df = dao.getObjects()


Final = dataExtractor.get_final_df(df)
X = Final.iloc[:, :-1].values
y = Final.iloc[:, -1].values

print("-" * 50)
print("Chosen features :    " + str(list(Final.columns)))
print("-" * 25 + " Mutual Information Data " + "-" * 25 )
feature_names = Final.columns
res = mutual_info_classif(X, y)
print(dict(zip(feature_names, res)))
print("-" * 50)
print( "Number of Objects with constraints: " +str(Final[Final['ContainsConstraints'] == 1 ].shape[0]))
print( "Number of Objects without constraints : " +str(Final[Final['ContainsConstraints'] == 0 ].shape[0]))

# for equal for target variable

if sampling_strategy == 'under':
    #Under-sample the majority
    sampler = RandomUnderSampler()
else:
    #over-sample the minority
    sampler = RandomOverSampler()

X, y = sampler.fit_resample(X, y)

# #Split to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=0)

print("-" * 25 + " Sampling Data " + "-" * 25 )
print( "Number of Rows in X after sampling is: " +str(X.shape[0]))
print( "Number of Rows in y after sampling is : " +str(y.shape[0]))

print("-" * 25 + " Data stats " + "-" * 25 )
print( "Number of Rows in Train Set is : " +str(X_train.shape[0]))
print( "Number of Rows in Test Set is : " +str(X_test.shape[0]))

print("-" * 25 + " Results " + "-" * 25 )
classify(X_train, X_test, y_train, y_test)

sys.stdout.close()
#featureImportance(feature_names)