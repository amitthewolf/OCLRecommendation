## Importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
import sqlite3
import pandas as pd
import numpy as np
from time import time
from scipy.stats import entropy
import numpy as np
from datetime import datetime


def information_gain(X, y):

    def _entropy(labels):
        counts = np.bincount(labels)
        return entropy(counts, base=None)

    def _ig(x, y):
        # indices where x is set/not set
        x_set = np.nonzero(x)[0]
        x_not_set = np.delete(np.arange(x.shape[0]), x_set)

        h_x_set = _entropy(y[x_set])
        h_x_not_set = _entropy(y[x_not_set])

        return entropy_full - (((len(x_set) / f_size) * h_x_set)
                             + ((len(x_not_set) / f_size) * h_x_not_set))

    entropy_full = _entropy(y)

    f_size = float(X.shape[0])

    scores = np.array([_ig(x, y) for x in X.T])
    return scores


#functions
def CheckifConstraint(genre):
    if genre == 0:
        return 0
    else:
        return 1

def createBalancedData():
    NoCons_indices = df[df.ContainsConstraints == 0].index
    Cons_indices = df[df.ContainsConstraints == 1].index
    # random_NoCons = np.random.choice(NoCons_indices, 18164, replace=False)
    random_NoCons = np.random.choice(NoCons_indices, 9082, replace=False)
    RandomNoCons_sample = df.loc[random_NoCons]
    Constraints_sample = df.loc[Cons_indices]
    return pd.concat([RandomNoCons_sample,Constraints_sample],ignore_index=True)

conn = sqlite3.connect("ThreeEyesDB.db")
df = pd.read_sql("SELECT * FROM Objects", conn)

#Target Value Column
df['ContainsConstraints'] = df.apply(lambda x: CheckifConstraint(x['ConstraintsNum']), axis=1)

#Create Balanced Dataframe 1:2 ratio / 1:1 ratio
Final = createBalancedData()
Final = Final.drop('FileLocation', axis=1)
Final = Final.drop('ObjectName', axis=1)
Final = Final.drop('ModelName', axis=1)
Final = Final.drop('LastRelationID', axis=1)
Final = Final.drop('SemanticWords', axis=1)
Final = Final.drop('ObjectID', axis=1)
Final = Final.drop('ConstraintsNum', axis=1)

X = Final.iloc[:, :-1].values
y = Final.iloc[:, -1].values

#Split to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


print("--------------------GNB model----------------------")
gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print("GNB Accuracy = " + str(accuracy_score(y_test, preds)))
#
print("--------------------KNeighbors model----------------")
KNN = KNeighborsClassifier()
KNN_model = KNN.fit(X_train, y_train)
KNN_preds = KNN.predict(X_test)
print("KNN Accuracy = " + str(accuracy_score(y_test, KNN_preds)))


print("--------------------Random Forest model---------------")
RF = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
RF_model = RF.fit(X_train, y_train)
RF_preds = RF.predict(X_test)
print("Random Forest Accuracy = " + str(accuracy_score(y_test, RF_preds)))


print("--------------------Information Gain----------------------")

res = information_gain(X_train,y_train)
print(res)

