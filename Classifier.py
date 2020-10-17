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

def CalcGNBPrecision(Train_X,Train_Y,Test_X,Test_Y):
    gnb = GaussianNB()
    gnb_model = gnb.fit(Train_X, Train_Y)
    preds = gnb.predict(Test_X)
    GNB = accuracy_score(Test_Y, preds)
    return GNB

def CalcKNNPrecision(Train_X,Train_Y,Test_X,Test_Y):
    KNN = KNeighborsClassifier()
    KNN_model = KNN.fit(Train_X, Train_Y)
    KNN_preds = KNN.predict(Test_X)
    KNN = accuracy_score(Test_Y, KNN_preds)
    return KNN

def GetTestData(ModelName):
    conn = sqlite3.connect("ThreeEyesDB.db")
    print
    Test = pd.read_sql("SELECT * FROM Objects WHERE ModelName = "+"'"+str(ModelName)+"'", conn)
    return Test

def GetTrainData(ModelName):
    conn = sqlite3.connect("ThreeEyesDB.db")
    print
    Train = pd.read_sql("SELECT * FROM Objects WHERE ModelName <> " +"'"+str(ModelName)+"'", conn)
    return Train

def DropFeatures(Dataframe):
    Dataframe = Dataframe.drop('FileLocation', axis=1)
    Dataframe = Dataframe.drop('ObjectName', axis=1)
    Dataframe = Dataframe.drop('ModelName', axis=1)
    Dataframe = Dataframe.drop('LastRelationID', axis=1)
    Dataframe = Dataframe.drop('SemanticWords', axis=1)
    Dataframe = Dataframe.drop('ObjectID', axis=1)
    return Dataframe.drop('ConstraintsNum', axis=1)


def GetX(Dataframe):
    return Dataframe.iloc[:, :-1].values

def GetY(Dataframe):
    return Dataframe.iloc[:, -1].values

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




# print("-----------------train data-----------------------")

# Feature Selection

# print(X_train)
# print(y_train)

print("--------------------Information Gain----------------------")

res = information_gain(X_train,y_train)
print(res)


# print(CalcPrecision(X_train,y_train,X_test,y_test))








             # Group Cross-Validation ---------------------------------------------------------------------------
# conn = sqlite3.connect('ThreeEyesDB.db')
# c = conn.cursor()
# c.execute("SELECT DISTINCT ModelName FROM Objects")
# conn.commit()
# result = c.fetchall()
# Counter = 0
# GNBScore = 0
# KNNScore = 0
# time = datetime.now()
#

# for ModelName in result:
#     Counter += 1
#     print("ModelName - " + str(ModelName[0]))
#     Train = DropFeatures(GetTrainData(ModelName[0]))
#     Test = DropFeatures(GetTestData(ModelName[0]))
#     X_Train = GetX(Train)
#     Y_Train = GetY(Train)
#     X_Test = GetX(Test)
#     Y_Test = GetY(Test)
#     GNBScore += float(CalcGNBPrecision(X_Train, Y_Train, X_Test, Y_Test))
#     KNNScore += float(CalcKNNPrecision(X_Train, Y_Train, X_Test, Y_Test))
#     print(Counter)
#     print(datetime.now() - time)
# print("Counter = "+str(Counter))
# print("GNB - "+str(GNBScore/Counter))
# print("KNN - "+str(KNNScore/Counter))

            # Group Cross-Validation ---------------------------------------------------------------------------




















# ## Importing required libraries
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import sqlite3
# import pandas as pd
# import numpy as np
# from time import time
#
# #functions
# def CheckifConstraint(genre):
#     if genre == 0:
#         return 0
#     else:
#         return 1
#
#
# start = time()
# conn = sqlite3.connect("TestDB.db")
# df = pd.read_sql("SELECT * FROM Objects", conn)
#
# #New Column containing boolean value
# df['ContainsConstraints'] = df.apply(lambda x: CheckifConstraint(x['ConstraintsNum']), axis=1)
#
# #Create Balanced Dataframe 1:2 ratio / 1:1 ratio
#
# NoCons_indices = df[df.ContainsConstraints == 0].index
# Cons_indices = df[df.ContainsConstraints == 1].index
# random_NoCons = np.random.choice(NoCons_indices, 18164, replace=False)
# # random_NoCons = np.random.choice(NoCons_indices, 9082, replace=False)
# RandomNoCons_sample = df.loc[random_NoCons]
# Constraints_sample = df.loc[Cons_indices]
# Final = pd.concat([RandomNoCons_sample,Constraints_sample],ignore_index=True)
#
# #Split to Train and Test 0.7/0.3
#
# # NewFinal = pd.DataFrame(np.random.randn(18164, 2))
# NewFinal = pd.DataFrame(np.random.randn(27246, 2))
# msk = np.random.rand(len(NewFinal)) < 0.7
# train = Final[msk]
# test = Final[~msk]
# print(len(train))
# print(len(test))
#
#
# # create train Feables
#
# labels = train.loc[:, 'ContainsConstraints']
# features = train.drop('ContainsConstraints', axis=1)
# features = features.drop('FileLocation', axis=1)
# features = features.drop('ObjectName', axis=1)
# features = features.drop('LastRelationID', axis=1)
# features = features.drop('SemanticWords', axis=1)
# features = features.drop('ObjectID', axis=1)
# features = features.drop('ConstraintsNum', axis=1)
# print(features)
#
# # create test Feables
#
# test_labels = test.loc[:, 'ContainsConstraints']
# test_features = test.drop('ContainsConstraints', axis=1)
# test_features = test_features.drop('FileLocation', axis=1)
# test_features = test_features.drop('ObjectName', axis=1)
# test_features = test_features.drop('LastRelationID', axis=1)
# test_features = test_features.drop('SemanticWords', axis=1)
# test_features = test_features.drop('ObjectID', axis=1)
# test_features = test_features.drop('ConstraintsNum', axis=1)
# print(test_features)
#
#
# # models :
#
# print("--------------------starting building GNB model----------------------")
# gnb = GaussianNB()
# gnb_model = gnb.fit(features, labels)
# preds = gnb.predict(test_features)
# print("GNB Accuracy = "+str(accuracy_score(test_labels, preds)))
# print("-------------------done building GNB model-------------------")
#
# print("--------------------starting building KNeighbors model----------------------")
# KNN = KNeighborsClassifier()
# KNN_model = KNN.fit(features, labels)
# KNN_preds = KNN.predict(test_features)
# print("KNN Accuracy = "+str(accuracy_score(test_labels, KNN_preds)))
# print("-------------------done building KNeighbors model-------------------")
#
# print("--------------------starting building Random Forest model----------------------")
# RF = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
# RF_model = RF.fit(features, labels)
# RF_preds = RF.predict(test_features)
# print("Random Forest Accuracy = "+str(accuracy_score(test_labels, RF_preds)))
# print("-------------------done building Random Forest model-------------------")
#
#
#


