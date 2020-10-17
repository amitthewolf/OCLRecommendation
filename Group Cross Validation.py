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


conn = sqlite3.connect('ThreeEyesDB.db')
c = conn.cursor()
c.execute("SELECT DISTINCT ModelName FROM Objects")
conn.commit()
result = c.fetchall()
Counter = 0
GNBScore = 0
KNNScore = 0
time = datetime.now()


for ModelName in result:
    Counter += 1
    print("ModelName - " + str(ModelName[0]))
    Train = DropFeatures(GetTrainData(ModelName[0]))
    Test = DropFeatures(GetTestData(ModelName[0]))
    X_Train = GetX(Train)
    Y_Train = GetY(Train)
    X_Test = GetX(Test)
    Y_Test = GetY(Test)
    GNBScore += float(CalcGNBPrecision(X_Train, Y_Train, X_Test, Y_Test))
    KNNScore += float(CalcKNNPrecision(X_Train, Y_Train, X_Test, Y_Test))
    print(Counter)
    print(datetime.now() - time)
print("Counter = "+str(Counter))
print("GNB - "+str(GNBScore/Counter))
print("KNN - "+str(KNNScore/Counter))
















