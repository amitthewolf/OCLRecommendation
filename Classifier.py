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
from DAO import DAO
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# #functions
def CheckifConstraint(genre):
    if genre == 0:
        return 0
    else:
        return 1

def CheckifReferenced(genre):
    if not genre or genre == 'nan' or genre == 0:
        return 0
    else:
        return 1

def inherits_column(value):
    if not value or value == 0 or value == "":
        return 0
    else:
        return 1

#
# def createBalancedData():
#     NoCons_indices = df[df.ContainsConstraints == 0].index
#     Cons_indices = df[df.ContainsConstraints == 1].index
#     random_NoCons = np.random.choice(NoCons_indices, 3005, replace=False)
#     RandomNoCons_sample = df.loc[random_NoCons]
#     Constraints_sample = df.loc[Cons_indices]
#     return pd.concat([RandomNoCons_sample, Constraints_sample], ignore_index=True)
#
# def createBalancedDataRef():
#     NoCons_indices = df[df.Referenced == 0].index
#     Cons_indices = df[df.Referenced == 1].index
#     # random_NoCons = np.random.choice(NoCons_indices, 3572, replace=False)
#     random_NoCons = np.random.choice(NoCons_indices, 10000, replace=False)
#     RandomNoCons_sample = df.loc[random_NoCons]
#     Constraints_sample = df.loc[Cons_indices]
#     return pd.concat([RandomNoCons_sample, Constraints_sample], ignore_index=True)

def drop_columns(Final):
    Final = Final.drop('ModelID', axis=1)
    Final = Final.drop('ObjectName', axis=1)
    Final = Final.drop('ModelName', axis=1)
    Final = Final.drop('LastRelationID', axis=1)
    Final = Final.drop('SemanticWords', axis=1)
    Final = Final.drop('ObjectID', axis=1)
    Final = Final.drop('ConstraintsNum', axis=1)
    Final = Final.drop('properties_names', axis=1)
    Final = Final.drop('inheriting_from', axis=1)
    # Final = Final.drop('ReferencedInConstraint', axis=1)
    # Final = Final.drop('is_abstract', axis=1)
    # Final = Final.drop('inherits', axis=1)
    return Final

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


def classify():
    models = [ GaussianNB(),KNeighborsClassifier(),RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced') ]
    for model in models:
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        print(model.__class__.__name__ + " : ")
        print(" Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
        print(" Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))
        print( '*' * 50)


DAO = DAO()
df = DAO.getObjects()

df['inherits'] = df.apply(lambda x: inherits_column(x['inheriting_from']), axis=1)

# Target Value Column
df['ContainsConstraints'] = df.apply(lambda x: CheckifConstraint(x['ConstraintsNum']), axis=1)
# df['Referenced'] = df['ReferencedInConstraint'].fillna(0)

print(df[df['ContainsConstraints'] == 1 ].shape)
Final = df
Final = drop_columns(Final)
X = Final.iloc[:, :-1].values
y = Final.iloc[:, -1].values
print()
print("--------------------Mutual Information----------------------")
feature_names = Final.columns
res = mutual_info_classif(X, y)
print(dict(zip(feature_names, res)))

# equal ratio of 0 and 1 for target variable
oversample = RandomUnderSampler()

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
#
# #Split to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.25, random_state=0)

print("--------------------Under Sampling----------------------")
print( "Number of Rows in Train Set is : " +str(X_train.shape))
print( "Number of Rows in Test Set is : " +str(X_test.shape))

print("--------------------Results----------------------")
classify()

featureImportance(feature_names)
