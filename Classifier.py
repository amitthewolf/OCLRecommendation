## Importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import sqlite3
import pandas as pd
import numpy as np
from time import time
from scipy.stats import entropy
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime


# #functions
def CheckifConstraint(genre):
    if genre == 0:
        return 0
    else:
        return 1
#
def createBalancedData():
    NoCons_indices = df[df.ContainsConstraints == 0].index
    Cons_indices = df[df.ContainsConstraints == 1].index
    # random_NoCons = np.random.choice(NoCons_indices, 18164, replace=False)
    random_NoCons = np.random.choice(NoCons_indices, 9082, replace=False)
    RandomNoCons_sample = df.loc[random_NoCons]
    Constraints_sample = df.loc[Cons_indices]
    return pd.concat([RandomNoCons_sample,Constraints_sample],ignore_index=True)

time = datetime.now()
conn = sqlite3.connect("ThreeEyesDB.db")
df = pd.read_sql("SELECT * FROM Objects", conn)

#Target Value Column
df['ContainsConstraints'] = df.apply(lambda x: CheckifConstraint(x['ConstraintsNum']), axis=1)

#Create Balanced Dataframe 1:2 ratio / 1:1 ratio
# Final = createBalancedData()
Final = df
Final = Final.drop('FileLocation', axis=1)
Final = Final.drop('ObjectName', axis=1)
Final = Final.drop('ModelName', axis=1)
Final = Final.drop('LastRelationID', axis=1)
Final = Final.drop('SemanticWords', axis=1)
Final = Final.drop('ObjectID', axis=1)
Final = Final.drop('ConstraintsNum', axis=1)

X = Final.iloc[:, :-1].values
y = Final.iloc[:, -1].values

print("--------------------Mutual Information----------------------")

from sklearn.feature_selection import mutual_info_classif
res = mutual_info_classif(X, y, discrete_features=True)
print(res)
print(datetime.now() - time)

print("--------------------Data Prep----------------------")
oversample = RandomOverSampler(sampling_strategy='minority')
# oversample = RandomOverSampler(sampling_strategy=0.5)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(len(y_over))

#
# #Split to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size = 0.25, random_state = 0)



print("--------------------GNB model----------------------")
gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print("GNB Accuracy = " + str(accuracy_score(y_test, preds)))
#
cm = confusion_matrix(y_test,preds)
cr = classification_report(y_test,preds)
print(cm)
print(cr)
print(datetime.now() - time)
print("--------------------KNeighbors model----------------")
KNN = KNeighborsClassifier()
KNN_model = KNN.fit(X_train, y_train)
KNN_preds = KNN.predict(X_test)
print("KNN Accuracy = " + str(accuracy_score(y_test, KNN_preds)))
cm = confusion_matrix(y_test,KNN_preds)
print(cm)
cr = classification_report(y_test,KNN_preds)
print(cr)
print(datetime.now() - time)
print("--------------------Random Forest model---------------")
RF = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
RF_model = RF.fit(X_train, y_train)
RF_preds = RF.predict(X_test)
print("Random Forest Accuracy = " + str(accuracy_score(y_test, RF_preds)))
cm = confusion_matrix(y_test,RF_preds)
print(cm)
cr = classification_report(y_test,KNN_preds)
print(cr)
print(datetime.now() - time)

