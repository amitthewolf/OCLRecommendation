from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
import sqlite3
import pandas as pd
from time import time
from scipy.stats import entropy
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime
from DAO import DAO

dao = DAO()

# dao.addColumnToModels()
conn = sqlite3.connect("ThreeEyesDB.db")
c = conn.cursor()

df = pd.read_sql("SELECT * FROM Models", conn)
LargestModel = dao.getLargestModel()


def Normalize(largestModel, oclConstraints, modelID):
    Normalized = oclConstraints / largestModel
    c.execute("UPDATE Models SET NormConstraints = ? WHERE ModelID = ?",(Normalized,modelID))
    conn.commit()
    return Normalized

# Target Value Column
df['NormConstraints'] = df.apply(lambda x: Normalize(LargestModel, x['ConstraintsNum'], x['ModelID']), axis=1)
print(df)

# print(df['NormConstraints'])