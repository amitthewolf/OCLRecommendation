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
    # c.execute()
    return oclConstraints/largestModel

# Target Value Column
df['NormConstraints'] = df.apply(lambda x: Normalize(LargestModel, x['ConstraintsNum'], x['ModelID']), axis=1)
print(df)

cols = "`,`".join([str(i) for i in df.columns.tolist()])
for i,row in df.iterrows():
    sql = "INSERT INTO `book_details` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
    c.execute(sql, tuple(row))
# c.execute("ALTER TABLE Models "
#           )
# print(df['NormConstraints'])