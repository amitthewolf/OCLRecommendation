## Importing required libraries
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from configparser import ConfigParser
from DAO import DAO
from Classification.TestConfig import TestConfig
from DataExtraction.dataExtractor import dataExtractor
from sklearn.model_selection import cross_val_score
from Classification.Logger import Logger
from Classification.PairClassifier import PairClassifier
from Classification.OperatorClassifier import OperatorClassifier
from Classification.Sampler import Sampler

ObjectIDInOrder = None
ModelIDInOrder = None


def classify(X_train, X_test, y_train, y_test):
    models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    for model in models:
        try:
            print(model.__class__.__name__ + " : ")
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            train_preds = model.predict(X_train)
            # print(" Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
            # print(" Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))

            clf = get_best_params(model, X_train, X_test, y_train, y_test, 'f1')  # added recently (f1\accuracy)

            print()
            if test_config.method != 'operator':
                scores = cross_val_score(model, X_train, y_train, cv=test_config.cross_val_k)
                # print("Cross-Validation result for k = {} : ".format(test_config.cross_val_k))
                # print('Scores :  {} '.format(scores))
                print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
                print('-' * 50)

                # added recently - here
                other_scores = cross_val_score(clf, X_train, y_train, cv=test_config.cross_val_k)
                print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (other_scores.mean(), other_scores.std()))
                print('-' * 50)
                LogSamples(model.__class__.__name__, X_test, y_test, test_preds, test_config, ObjectIDInOrder, ModelIDInOrder)
                logger.LogResult(model.__class__.__name__, y_test, test_preds, y_train, train_preds, scores,test_config, featureNames, iterations, fixed_section)
            else:
                Trainscore, TestScore = op_clf.ScoreOperator(train_preds, test_preds, y_train, y_test)
                logger.LogResultOperator(model.__class__.__name__, Trainscore, TestScore, test_config, featureNames, iterations, fixed_section)
        except Exception as e:
            print(e)
            print(" Invalid Y Dimentions")

def get_best_params(model, X_train, X_test, y_train, y_test, score):
    print("Tuning Hyper-Parameters ... ")
    x = model.__class__.__name__
    if x == 'RandomForestClassifier':
        y = list(np.arange(50, 100, 200))
        param_grid_rf = {
            'n_estimators': y,
            'max_features': ['auto', 'sqrt', 'log2'],
            'criterion' : ["gini", "entropy"]
        }
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=5, scoring=score)
        CV_rfc.fit(X_train, y_train)
        return CV_rfc

    elif x == 'GaussianNB':
        params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
        gs_NB = GridSearchCV(estimator=model, param_grid=params_NB, cv=5, scoring=score)
        gs_NB.fit(X_train, y_train)
        return gs_NB

    elif x == 'KNeighborsClassifier':
        sqrt_samples = math.sqrt(X_train.shape[0])
        y = np.arange(1, sqrt_samples, 2)
        arr = list(np.nan_to_num(y, copy=False).astype(np.int))
        # 'metrics': ['minkowski', 'euclidean', 'manhattan'],
        params = {'weights': ['uniform', 'distance'], 'n_neighbors': arr}
        clf = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring=score)
        clf.fit(X_train, y_train)
        return clf

def LogSamples(modelName, XTest, YTest, PredTest, test_config,ObjectIDInOrder= None, ModelIDInOrder= None):
    FPCounter = 0
    FNCounter = 0
    for index in range(len(PredTest)):
        if PredTest[index] == 1 and YTest.array[index] == 0 and FPCounter < 3:
            FPCounter += 1
            SampleToLog = XTest.iloc[index]
            Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
            Log_DF['Model'] = modelName
            if test_config.method != 'pairs':
                ObjectIDRow = ObjectIDInOrder.iloc[index]
                CurrObjectID = ObjectIDRow.item()
                ModelRow = dao.getModelRowByObjectID(CurrObjectID)
                ModelList = list(ModelRow[0])
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            else:
                ModelIDRow = ModelIDInOrder.iloc[index]
                CurrModelID = ModelIDRow.item()
                ModelRow = dao.getSpecificModel(CurrModelID)
                ModelList = list(ModelRow)
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            logger.LogSamples(Log_DF, 'FP', index=False)
        elif PredTest[index] == 0 and YTest.array[index] == 1 and FNCounter < 3:
            FNCounter += 1
            SampleToLog = XTest.iloc[index]
            Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
            Log_DF['Model'] = modelName
            if test_config.method != 'pairs':
                ObjectIDRow = ObjectIDInOrder.iloc[index]
                CurrObjectID = ObjectIDRow.item()
                ModelRow = dao.getModelRowByObjectID(CurrObjectID)
                ModelList = list(ModelRow[0])
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            else:
                ModelIDRow = ModelIDInOrder.iloc[index]
                CurrModelID = ModelIDRow.item()
                ModelRow = dao.getSpecificModel(CurrModelID)
                ModelList = list(ModelRow)
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            logger.LogSamples(Log_DF, 'FN', index=False)
        if FNCounter == 3 and FPCounter == 3:
            return

def getOperatorY(ObjectIDs):
    return dao.GetConstraintRandomSample(ObjectIDs)

def run(df, test_config):
    X = df.loc[:, df.columns != test_config.target]
    X = X.loc[:, X.columns != "ObjectID"]
    ObjectIDInOrder = df["ObjectID"]

    print("Test Statistics For :", test_method)
    print("-" * 50)
    print("Dataframe columns :    " + str(list(X.columns)))
    print("-" * 50)

    feature_names = X.columns
    if test_config.method == 'operator':
        y = getOperatorY(df["ObjectID"])
    else:
        y = df[test_config.target]
        print("Number of positive records : " + str(df[df[test_config.target] == 1].shape[0]))
        print("Number of negative records : " + str(df[df[test_config.target] == 0].shape[0]))
        print("-" * 25 + "Mutual Information" + "-" * 25)
        res = mutual_info_classif(X, y)
        mi_dict = dict(zip(feature_names, res))
        print(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))

    # #Split to Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)

    print("-" * 25 + " Data stats " + "-" * 25)
    print("Number of Rows in Train Set is : " + str(X_train.shape[0]))
    print("Number of Rows in Test Set is : " + str(X_test.shape[0]))

    print(datetime.now())

    print("-" * 25 + " Results " + "-" * 25)
    classify(X_train, X_test, y_train, y_test)




dao = DAO()
config = ConfigParser()
dataExtractor = dataExtractor()
logger = Logger()
op_clf = OperatorClassifier()
# sampler = Sampler(None, None)

# get configurations
config.read('conf.ini')
fixed_section = config['fixed_params']
iterations = int(fixed_section['iterations'])
random_param_sampling = fixed_section['random']
test_method = fixed_section['test_method']

models_number = dao.get_models_number()
graphlets_flag = fixed_section['graphlets_flag']

if random_param_sampling == 'True':
    test_config = TestConfig(graphlets_flag, models_number, test_method)
else:
    test_config = TestConfig(graphlets_flag, models_number, test_method, random=False)


for i in range(iterations):
    featureNames = test_config.classifier_section['featureNames'].split(',')
    df = dao.getObjects()
    test_config.update_iteration_params(i)
    if test_method != 'pairs':
        df = dataExtractor.get_final_df(df, featureNames, test_config)
        df.to_csv("fd.csv")
        ObjectIDInOrder = df['ObjectID']
        run(df, test_config)
    elif test_method == 'pairs':
        b_df, ub_df, ModelIDInOrder = dataExtractor.get_final_df(df, featureNames, test_config)
        pairs_clf = PairClassifier.getInstance(test_config)
        pairs_clf.predict(b_df, ub_df)