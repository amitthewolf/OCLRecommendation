## Importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import sqlite3
import pandas as pd
import numpy as np
from time import time

#functions
def CheckifConstraint(genre):
    if genre == 0:
        return 0
    else:
        return 1

def createBalancedData():
    NoCons_indices = df[df.ContainsConstraints == 0].index
    Cons_indices = df[df.ContainsConstraints == 1].index
    random_NoCons = np.random.choice(NoCons_indices, 9082, replace=False)
    RandomNoCons_sample = df.loc[random_NoCons]
    Constraints_sample = df.loc[Cons_indices]
    return pd.concat([RandomNoCons_sample,Constraints_sample],ignore_index=True)

conn = sqlite3.connect("TestDB.db")
df = pd.read_sql("SELECT * FROM Objects", conn)

#Target Value Column
df['ContainsConstraints'] = df.apply(lambda x: CheckifConstraint(x['ConstraintsNum']), axis=1)

#Create Balanced Dataframe 1:2 ratio / 1:1 ratio
Final = createBalancedData()

X = Final.iloc[:, :-1].values
y = Final.iloc[:, -1].values

#Split to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print("-----------------train data-----------------------")

# Feature Selection
y_train = Final.loc[:, 'ContainsConstraints']
X_train = Final.drop('ContainsConstraints', axis=1)
X_train = X_train.drop('FileLocation', axis=1)
X_train = X_train.drop('ObjectName', axis=1)
X_train = X_train.drop('LastRelationID', axis=1)
X_train = X_train.drop('SemanticWords', axis=1)
X_train = X_train.drop('ObjectID', axis=1)
X_train = X_train.drop('ConstraintsNum', axis=1)
print(X_train)

y_test = Final.loc[:, 'ContainsConstraints']
X_test = Final.drop('ContainsConstraints', axis=1)
X_test = X_test.drop('FileLocation', axis=1)
X_test = X_test.drop('ObjectName', axis=1)
X_test = X_test.drop('LastRelationID', axis=1)
X_test = X_test.drop('SemanticWords', axis=1)
X_test = X_test.drop('ObjectID', axis=1)
X_test = X_test.drop('ConstraintsNum', axis=1)
print(X_test)


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

