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
from node2vec import node2vec
from sklearn.model_selection import cross_val_score

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


# df = dao.getObjects()
config = ConfigParser()
dataExtractor = DataExtractor()

#get configurations
config.read('conf.ini')
conf = config['classifier']
sampling_strategy = conf['sampling']
featuresNames = conf['featureNames']
target = conf['Target']
iterations = conf['iterations']
# cross_val_flag = conf['cross_val']
test_ratio = float(conf['test_ratio'])

featuresNames = featuresNames.split(',')

conf = config['node2vec']
n2v = conf['Node2Vec']
num_n2v = conf['features_num']
if n2v == 'True':
    node2vec = node2vec(num_n2v)
    n2v_features = ['N2V_' + str(i) for i in range(1, int(num_n2v)+1)]
    featuresNames = featuresNames + n2v_features

target = target.split(',')

dao = DAO()
df = dao.getObjects()

df = dataExtractor.get_final_df(df,featuresNames,target)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("-" * 50)
print("Chosen features :    " + str(list(df.columns)))
print("-" * 50)
print( "Number of Objects with constraints: " + str(df[df['ContainsConstraints'] == 1].shape[0]))
print( "Number of Objects without constraints : " + str(df[df['ContainsConstraints'] == 0].shape[0]))

# for equal for target variable

if sampling_strategy == 'under':
    #Under-sample the majority
    sampler = RandomUnderSampler()
else:
    #over-sample the minority
    sampler = RandomOverSampler()

X, y = sampler.fit_resample(X, y)

print("-" * 25 + "Mutual Information" + "-" * 25 )
feature_names = df.columns
res = mutual_info_classif(X, y)
print(dict(zip(feature_names, res)))

# #Split to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_ratio, random_state=0)

print("-" * 25 + " Sampling Data " + "-" * 25 )
print( "Number of Rows in Data after sampling is: " +str(X.shape[0]))

print("-" * 25 + " Data stats " + "-" * 25 )
print( "Number of Rows in Train Set is : " +str(X_train.shape[0]))
print( "Number of Rows in Test Set is : " +str(X_test.shape[0]))


clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


sys.stdout = open('outputs/outputs.txt', 'a')
print("#" * 30 + " New Experiment " + "#" * 30 )
print(datetime.now())
# str1 = ''.join(featuresNames)
# print("features: "+ ''.join(featuresNames))
print("features: "+str(list(df.columns)))
print("iterations: "+iterations)
print("sampling strategy: "+sampling_strategy)
print("-" * 25 + " Results " + "-" * 25 )
classify(X_train, X_test, y_train, y_test)
sys.stdout.close()