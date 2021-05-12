## Importing required libraries
import math

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
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
import statistics

ModelIDInOrder = None

def classify(X_train, X_test, y_train, y_test,TestObjectIDInOrder=None):
    models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    for model in models:
        if model.__class__.__name__ == 'RandomForestClassifier':
            try:
                print(model.__class__.__name__ + " : ")
                # "Regular" classifiers
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
                    print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (
                        scores.mean(), scores.std()))
                    print('-' * 50)

                    # added recently - here
                    other_scores = cross_val_score(clf, X_train, y_train, cv=test_config.cross_val_k)
                    print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (
                        other_scores.mean(), other_scores.std()))
                    print('-' * 50)
                    # to here

                    LogSamples(model.__class__.__name__, X_test, y_test, test_preds,TestObjectIDInOrder)
                    LogResult(model.__class__.__name__, y_test, test_preds, y_train, train_preds, scores)
                else:
                    Trainscore, TestScore, Fscore = ScoreOperator(train_preds, test_preds, y_train, y_test)
                    LogResultOperator(model.__class__.__name__, Trainscore, TestScore,Fscore)
            except Exception as e:
                print(e)
                print(" Invalid Y Dimentions")


def ScoreOperator(train_preds, test_preds, y_train, y_test):
    RocAucScores = []
    for index in range(len(y_train)):
        try:
            RocAucScores.append(roc_auc_score(y_train[index], train_preds[index]))
        except:
            pass
    SumScores = sum(RocAucScores)
    TrainFinalScore = SumScores / len(RocAucScores)
    print(" Train Roc Auc Score :", TrainFinalScore)
    m = MultiLabelBinarizer().fit(y_train)

    FScore = f1_score(m.transform(y_train),
             m.transform(train_preds),
             average='macro')
    print(" Train F-Score :", FScore)
    RocAucScores = []
    for index in range(len(y_test)):
        try:
            RocAucScores.append(roc_auc_score(y_test[index], test_preds[index]))
        except:
            pass
    SumScores = sum(RocAucScores)
    TestFinalScore = SumScores / len(RocAucScores)
    print(" Test Roc Auc Score :", TestFinalScore)
    m = MultiLabelBinarizer().fit(y_test)

    FScore = f1_score(m.transform(y_test),
                      m.transform(test_preds),
                      average='macro')
    print(" Test F-Score :", FScore)
    return TrainFinalScore, TestFinalScore, FScore


def predict_pairs(X_train, X_test, y_train, y_test):
    models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    results = {}

    for model in models:
        # print(model.__class__.__name__ + " : ")
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        results[model.__class__.__name__] = accuracy_score(y_test, test_preds)

    return results


def get_best_params(model, X_train, X_test, y_train, y_test, score):
    x = model.__class__.__name__
    if x == 'RandomForestClassifier':
        y = list(np.arange(100, 700, 100))
        param_grid_rf = {
            'n_estimators': y,
            'max_features': ['auto', 'sqrt', 'log2']
        }
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=5, scoring=score)
        CV_rfc.fit(X_train, y_train)
        return CV_rfc

    elif x == 'GaussianNB':
        params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
        gs_NB = GridSearchCV(estimator=model, param_grid=params_NB, cv=5, verbose=1, scoring=score)
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


def LogResult(modelName, YTest, PredTest, YTrain, PredTrain, scores):
    if test_config.n2v_flag == 'True':
        data = {'Timestamp': datetime.now(),
                'Target': test_config.target,
                'Method': test_config.method,
                'Model': modelName,
                'Features': ','.join(featureNames[:-1]),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
                'Graphlets': fixed_section['graphlets_flag'],
                'N2V_Flag': test_config.n2v_flag,
                'N2V_Features_Num': test_config.n2v_features_num,
                'N2V_use_Att': 'true',
                'N2V_use_Inhe': 'true',
                'N2V_ReturnWe': test_config.n2v_return_weight,
                'N2V_WalkLen': test_config.n2v_walklen,
                'N2V_Epochs': test_config.n2v_epochs,
                'N2V_NeighborWeight': test_config.n2v_neighbor_weight,
                'Num_PCA': test_config.pca,
                'Iterations': iterations,
                'Random': random_param_sampling,
                'Train Score': accuracy_score(YTrain, PredTrain),
                'Test Score': accuracy_score(YTest, PredTrain),
                'Mean': scores.mean(),
                'Std': scores.std(),
                }

        Log_DF = pd.DataFrame(data, columns=['Timestamp',
                                             'Target',
                                             'Method',
                                             'Model',
                                             'Features',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
                                             'Graphlets',
                                             'N2V_Flag',
                                             'N2V_Features_Num',
                                             'N2V_use_Att',
                                             'N2V_use_Inhe',
                                             'N2V_ReturnWe',
                                             'N2V_WalkLen',
                                             'N2V_Epochs',
                                             'N2V_NeighborWeight',
                                             'Num_PCA',
                                             'Iterations',
                                             'Random',
                                             'Train Score',
                                             'Test Score',
                                             'Mean',
                                             'Std'], index=[0])
    else:
        data = {'Timestamp': datetime.now(),
                'Target': test_config.target,
                'Method': test_config.method,
                'Model': modelName,
                'Features': ','.join(featureNames[:-1]),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
                'Graphlets': fixed_section['graphlets_flag'],
                'N2V_Flag': test_config.n2v_flag,
                'N2V_Features_Num': '-',
                'N2V_use_Att': '-',
                'N2V_use_Inhe': '-',
                'N2V_ReturnWe': '-',
                'N2V_WalkLen': '-',
                'N2V_Epochs': '-',
                'N2V_NeighborWeight': '-',
                'Num_PCA': '-',
                'Iterations': iterations,
                'Random': random_param_sampling,
                'Train Score': accuracy_score(YTrain, PredTrain),
                'Test Score': accuracy_score(YTest, PredTest),
                'Mean': scores.mean(),
                'Std': scores.std(),
                }

        Log_DF = pd.DataFrame(data, columns=['Timestamp',
                                             'Target',
                                             'Method',
                                             'Model',
                                             'Features',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
                                             'Graphlets',
                                             'N2V_Flag',
                                             'N2V_Features_Num',
                                             'N2V_use_Att',
                                             'N2V_use_Inhe',
                                             'N2V_ReturnWe',
                                             'N2V_WalkLen',
                                             'N2V_Epochs',
                                             'N2V_NeighborWeight',
                                             'Num_PCA',
                                             'Iterations',
                                             'Random',
                                             'Train Score',
                                             'Test Score',
                                             'Mean',
                                             'Std'], index=[0])
    log = Logger()
    log.append_df_to_excel(Log_DF, header=None, index=False)


def LogResultOperator(modelName, Train, Test,Fscore):
    if test_config.n2v_flag == 'True':
        data = {'Timestamp': datetime.now(),
                'Target': test_config.target,
                'Method': test_config.method,
                'Model': modelName,
                'Features': ','.join(featureNames[:-1]),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
                'Graphlets': fixed_section['graphlets_flag'],
                'N2V_Flag': test_config.n2v_flag,
                'N2V_Features_Num': test_config.n2v_features_num,
                'N2V_use_Att': 'true',
                'N2V_use_Inhe': 'true',
                'N2V_ReturnWe': test_config.n2v_return_weight,
                'N2V_WalkLen': test_config.n2v_walklen,
                'N2V_Epochs': test_config.n2v_epochs,
                'N2V_NeighborWeight': test_config.n2v_neighbor_weight,
                'Num_PCA': test_config.pca,
                'Iterations': iterations,
                'Random': random_param_sampling,
                'Train Score': Train,
                'Test Score': Test,
                'Fscore': Fscore,
                'Mean': '-',
                'Std': '-',
                }

        Log_DF = pd.DataFrame(data, columns=['Timestamp',
                                             'Target',
                                             'Method',
                                             'Model',
                                             'Features',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
                                             'Graphlets',
                                             'N2V_Flag',
                                             'N2V_Features_Num',
                                             'N2V_use_Att',
                                             'N2V_use_Inhe',
                                             'N2V_ReturnWe',
                                             'N2V_WalkLen',
                                             'N2V_Epochs',
                                             'N2V_NeighborWeight',
                                             'Num_PCA',
                                             'Iterations',
                                             'Random',
                                             'Train Score',
                                             'Test Score',
                                             'Fscore',
                                             'Mean',
                                             'Std'], index=[0])
    else:
        data = {'Timestamp': datetime.now(),
                'Target': test_config.target,
                'Method': test_config.method,
                'Model': modelName,
                'Features': ','.join(featureNames[:-1]),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
                'Graphlets': fixed_section['graphlets_flag'],
                'N2V_Flag': test_config.n2v_flag,
                'N2V_Features_Num': '-',
                'N2V_use_Att': '-',
                'N2V_use_Inhe': '-',
                'N2V_ReturnWe': '-',
                'N2V_WalkLen': '-',
                'N2V_Epochs': '-',
                'N2V_NeighborWeight': '-',
                'Num_PCA': '-',
                'Iterations': iterations,
                'Random': random_param_sampling,
                'Train Score': Train,
                'Test Score': Test,
                'Fscore': Fscore,
                'Mean': '-',
                'Std': '-',
                }

        Log_DF = pd.DataFrame(data, columns=['Timestamp',
                                             'Target',
                                             'Method',
                                             'Model',
                                             'Features',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
                                             'Graphlets',
                                             'N2V_Flag',
                                             'N2V_Features_Num',
                                             'N2V_use_Att',
                                             'N2V_use_Inhe',
                                             'N2V_ReturnWe',
                                             'N2V_WalkLen',
                                             'N2V_Epochs',
                                             'N2V_NeighborWeight',
                                             'Num_PCA',
                                             'Iterations',
                                             'Random',
                                             'Train Score',
                                             'Test Score',
                                             'Fscore',
                                             'Mean',
                                             'Std'], index=[0])
    log = Logger()
    log.append_df_to_excel(Log_DF, header=None, index=False)


def LogSamples(modelName, XTest, YTest, PredTest, TestObjectIDInOrder):
    FPCounter = 0
    FNCounter = 0
    for index in range(len(PredTest)) :
        if PredTest[index] == 1 and YTest.array[index] == 0 and modelName == 'RandomForestClassifier':
            FPCounter += 1
            SampleToLog = XTest.iloc[index]
            log = Logger()
            Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
            Log_DF['Model'] = modelName
            if test_config.method != 'pairs':
                ObjectIDRow = TestObjectIDInOrder.iloc[index]
                CurrObjectID = ObjectIDRow.item()
                ModelRow = dao.getModelRowByObjectID(CurrObjectID)
                ModelList = list(ModelRow[0])
                Log_DF['ObjectID'] = CurrObjectID
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            else:
                ModelIDRow = ModelIDInOrder.iloc[index]
                CurrModelID = ModelIDRow.item()
                ModelRow = dao.getSpecificModel(CurrModelID)
                ModelList = list(ModelRow)
                # Log_DF['ObjectID'] = CurrObjectID
                Log_DF['ModelID'] = ModelList[0]
                Log_DF['Path'] = ModelList[1]
                Log_DF['ConstraintsNum'] = ModelList[2]
                Log_DF['ObjectsNum'] = ModelList[3]
            log.LogSamples(Log_DF, 'FP', index=False)
        elif PredTest[index] == 0 and YTest.array[index] == 1 and modelName == 'RandomForestClassifier':
            FNCounter += 1
            SampleToLog = XTest.iloc[index]
            log = Logger()
            Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
            Log_DF['Model'] = modelName
            if test_config.method != 'pairs':
                ObjectIDRow = TestObjectIDInOrder.iloc[index]
                CurrObjectID = ObjectIDRow.item()
                ModelRow = dao.getModelRowByObjectID(CurrObjectID)
                ModelList = list(ModelRow[0])
                Log_DF['ObjectID'] = CurrObjectID
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
            log.LogSamples(Log_DF, 'FN', index=False)
        # if FNCounter == 3 and FPCounter == 3:
        #     return


def getOperatorY(ObjectIDs):
    return dao.GetConstraintRandomSample(ObjectIDs)


def run(df, test_config):
    X = df.loc[:, df.columns != test_config.target]


    print("Test Statistics :", test_method)
    print("-" * 50)
    print("Dataframe columns :    " + str(list(X.columns)))
    print("-" * 50)


    if test_config.method == 'operator':
        y = getOperatorY(df["ObjectID"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)
        TestObjectIDInOrder = X_test['ObjectID']
        X_train = X_train.loc[:, X.columns != "ObjectID"]
        X_test = X_test.loc[:, X.columns != "ObjectID"]
    else:
        y = df[test_config.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)
        TestObjectIDInOrder = X_test['ObjectID']
        X_train = X_train.loc[:, X.columns != "ObjectID"]
        X_test = X_test.loc[:, X.columns != "ObjectID"]
        feature_names = X.columns
        print("Number of positive records : " + str(df[df[test_config.target] == 1].shape[0]))
        print("Number of negative records : " + str(df[df[test_config.target] == 0].shape[0]))
        print("-" * 25 + "Mutual Information" + "-" * 25)
        res = mutual_info_classif(X, y)
        mi_dict = dict(zip(feature_names, res))
        print(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))

    # for equal for target variable
    # if test_config.sampling_strategy == 'under':
    #     # Under-sample the majority
    #     sampler = RandomUnderSampler()
    # else:
    #     # over-sample the minority
    #     sampler = RandomOverSampler()
    #
    # X, y = sampler.fit_resample(X, y)

    # #Split to Train and Test

    # print("-" * 25 + " Sampling Data " + "-" * 25)
    # print("Number of Rows in Data after sampling is: " + str(X.shape[0]))

    print("-" * 25 + " Data stats " + "-" * 25)
    print("Number of Rows in Train Set is : " + str(X_train.shape[0]))
    print("Number of Rows in Test Set is : " + str(X_test.shape[0]))
    print()

    print(datetime.now())
    # str1 = ''.join(featuresNames)
    # print("DataExtraction: "+ ''.join(featuresNames))
    # print("DataExtraction: " + str(list(df.columns)))

    # n2v_feat = "Features: " + n2v_section['n2v_features_num'] + ", Attributes: " + n2v_section['n2v_use_attributes'] + \
    #            ", Inheritance: " + n2v_section['n2v_use_inheritance'] + ", Return weight: " + \
    #            n2v_section['n2v_return_weight'] + ", Walklen: " + n2v_section['n2v_walklen'] + ", Epcochs:" + \
    #            n2v_section['n2v_epochs'] + ", Neighbour weight: " + n2v_section['n2v_neighbor_weight'] + \
    #            ", Use PCA: " + n2v_section['use_pca'] + ", PCA num: " + n2v_section['pca_num']
    #
    # if n2v_section['n2v_flag'] == 'True':
    #     print('Node2Vec Features:')
    #     print(n2v_feat)

    # ("sampling strategy: " + test_config.sampling_strategy)
    print("-" * 25 + " Results " + "-" * 25)
    classify(X_train, X_test, y_train, y_test,TestObjectIDInOrder)


# Train on balanced, Test on un-balanced
def prepare_pairs_test_train(bal_df, unbal_df):
    results = []
    models_ids = bal_df['ModelID'].unique()

    for model_id in models_ids:
        # filter relevant model
        unbal_df_model = unbal_df.loc[unbal_df['ModelID'] == model_id]
        unbal_df_model = unbal_df_model.drop("ModelID", axis=1)

        # filter all other models except the relevant one
        bal_df_models = bal_df.loc[bal_df['ModelID'] != model_id]
        bal_df_models = bal_df_models.drop("ModelID", axis=1)

        X_train = bal_df_models.loc[:, bal_df_models.columns != test_config.target]
        y_train = bal_df_models[test_config.target]

        X_test = unbal_df_model.loc[:, unbal_df_model.columns != test_config.target]
        y_test = unbal_df_model[test_config.target]

        results.append(predict_pairs(X_train, X_test, y_train, y_test))

    NB = []
    KNN = []
    RF = []

    for result_set in results:
        NB.append(result_set['GaussianNB'])
        KNN.append(result_set['KNeighborsClassifier'])
        RF.append(result_set['RandomForestClassifier'])

    print("Pairs Classification Results : \n \n ")

    print("Train on EACH balanced model and test over the rest of the un-balanced models: \n")
    print('GaussianNB ' + str(statistics.mean(NB)))
    print('KNeighborsClassifier ' + str(statistics.mean(KNN)))
    print('RandomForestClassifier ' + str(statistics.mean(RF)))
    print("-" * 50)

    print("Train on ALL balanced models and test over the rest of the un-balanced models accuracy: \n")

    bal_df_final = bal_df.drop("ModelID", axis=1)
    unbal_df_final = unbal_df.drop("ModelID", axis=1)
    X_train = bal_df_final.loc[:, bal_df_final.columns != test_config.target]
    y_train = bal_df_final[test_config.target]
    X_test = unbal_df_final.loc[:, unbal_df_final.columns != test_config.target]
    y_test = unbal_df_final[test_config.target]
    classify(X_train, X_test, y_train, y_test)


dao = DAO()
config = ConfigParser()
dataExtractor = dataExtractor()

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
    # print('*' * 50)
    # print("{} Experiment ".format(i + 1))
    # print('*' * 50)
    test_config.update_iteration_params(i)
    if test_method != 'pairs':
        df = dataExtractor.get_final_df(df, featureNames, test_config)
        run(df, test_config)
    elif test_method == 'pairs':
        b_df, ub_df, ModelIDInOrder = dataExtractor.get_final_df(df, featureNames, test_config)
        prepare_pairs_test_train(b_df, ub_df)
