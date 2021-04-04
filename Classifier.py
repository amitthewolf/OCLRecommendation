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
from sklearn import svm
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from dataExtractor import dataExtractor as DataExtractor
from DAO import DAO
from TestConfig import TestConfig
from node2vec import node2vec
from sklearn.model_selection import cross_val_score
from itertools import chain, combinations
from Logger import Logger


def classify(X_train, X_test, y_train, y_test):
    models = [GaussianNB(), KNeighborsClassifier(),RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')]

    for model in models:

        # "Regular" classifiers
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        print(model.__class__.__name__ + " : ")
        print(" Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
        print(" Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))
        print()
        # "Cross validated" classifiers
        scores = cross_val_score(model, X_train, y_train, cv=test_config.cross_val_k)
        print("Cross-Validation result for k = {} : ".format(test_config.cross_val_k))
        print('Scores :  {} '.format(scores))
        print("%0.2f average accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print('-' * 50)
        LogSamples(model.__class__.__name__, X_test, y_test, test_preds)
        LogResult(model.__class__.__name__,y_test, test_preds,y_train, train_preds,scores)


def LogResult(modelName, YTest,PredTest, YTrain, PredTrain,scores):
    if test_config.n2v_flag == 'True':
        data = {'Features': ','.join(featureNames[:-1]),
                'Timestamp': datetime.now(),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
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
                'Target': test_config.target,
                'Model': modelName,
                'Train Score': accuracy_score(y_train, train_preds),
                'Test Score': accuracy_score(y_test, test_preds),
                'Mean': scores.mean(),
                'Std': scores.std(),
                }

        Log_DF = pd.DataFrame(data, columns=['Features',
                                             'Timestamp',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
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
                                             'Target',
                                             'Model',
                                             'Train Score',
                                             'Test Score',
                                             'Mean',
                                             'Std'], index=[0])
        print(Log_DF)
    else:
        data = {'Features': ','.join(featureNames[:-1]),
                'Timestamp': datetime.now(),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
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
                'Target': test_config.target,
                'Model':modelName ,
                'Train Score': accuracy_score(YTrain, PredTrain),
                'Test Score': accuracy_score(YTest,PredTest),
                'Mean': scores.mean(),
                'Std': scores.std(),
                }

        Log_DF = pd.DataFrame(data, columns=['Features',
                                             'Timestamp',
                                             'Sampling',
                                             'Test_ratio',
                                             'Cross_Val_K',
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
                                             'Target',
                                             'Model',
                                             'Train Score',
                                             'Test Score',
                                             'Mean',
                                             'Std'], index=[0])
        print(Log_DF)
    log = Logger()
    log.append_df_to_excel(Log_DF, header=None, index=False)

def LogSamples(modelName, XTest, YTest, PredTest):
    Counter = 0
    for index in range(len(PredTest)):
        if(PredTest[index]==1 and YTest[index]==0):
            Counter += 1
            SampleToLog = XTest[index]
            dataset = pd.DataFrame({'Column1': SampleToLog[:, 0], 'Column2': SampleToLog[:, 1]})
            log = Logger()
            log.LogSamples(dataset, header=None, index=False)
            return


def run(test_config):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print("-" * 50)
    print("Chosen features :    " + str(list(df.columns)))
    print("-" * 50)
    print("Number of positive target records : " + str(df[df[test_config.target] == 1].shape[0]))
    print("Number of negative target records : " + str(df[df[test_config.target] == 0].shape[0]))

    # for equal for target variable
    if test_config.sampling_strategy == 'under':
        # Under-sample the majority
        sampler = RandomUnderSampler()
    else:
        # over-sample the minority
        sampler = RandomOverSampler()

    X, y = sampler.fit_resample(X, y)

    print("-" * 25 + "Mutual Information" + "-" * 25)
    feature_names = df.columns
    res = mutual_info_classif(X, y)
    print(dict(zip(feature_names, res)))

    # #Split to Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)

    print("-" * 25 + " Sampling Data " + "-" * 25)
    print("Number of Rows in Data after sampling is: " + str(X.shape[0]))

    print("-" * 25 + " Data stats " + "-" * 25)
    print("Number of Rows in Train Set is : " + str(X_train.shape[0]))
    print("Number of Rows in Test Set is : " + str(X_test.shape[0]))
    print()

    # sys.stdout = open('outputs/outputs.txt', 'a')

    print(datetime.now())
    # str1 = ''.join(featuresNames)
    # print("features: "+ ''.join(featuresNames))
    print("features: " + str(list(df.columns)))

    # n2v_feat = "Features: " + n2v_section['n2v_features_num'] + ", Attributes: " + n2v_section['n2v_use_attributes'] + \
    #            ", Inheritance: " + n2v_section['n2v_use_inheritance'] + ", Return weight: " + \
    #            n2v_section['n2v_return_weight'] + ", Walklen: " + n2v_section['n2v_walklen'] + ", Epcochs:" + \
    #            n2v_section['n2v_epochs'] + ", Neighbour weight: " + n2v_section['n2v_neighbor_weight'] + \
    #            ", Use PCA: " + n2v_section['use_pca'] + ", PCA num: " + n2v_section['pca_num']
    #
    # if n2v_section['n2v_flag'] == 'True':
    #     print('Node2Vec Features:')
    #     print(n2v_feat)

    print("sampling strategy: " + test_config.sampling_strategy)
    print("-" * 25 + " Results " + "-" * 25)
    classify(X_train, X_test, y_train, y_test)
    # sys.stdout.close()

dao = DAO()
config = ConfigParser()
dataExtractor = DataExtractor()



#get configurations
config.read('conf.ini')
fixed_section = config['fixed_params']
iterations = int(fixed_section['iterations'])
random_param_sampling = fixed_section['random']

if random_param_sampling == 'True':
    test_config = TestConfig()
else:
    test_config = TestConfig(random=False)




# ALL THE BELOW COMMENTED PRINTS FOR WRITING TO FILE LATER(AMIT)
for i in range(iterations):
    featureNames = test_config.classifier_section['featureNames'].split(',')
    df = dao.getObjects()
    print('*' * 50)
    print("{} Experiment ".format(i + 1))
    print('*' * 50)
    test_config.update_iteration_params(i)
    if test_config.n2v_flag == 'True':
        print("N2V process started ...")
        df = dataExtractor.Set_N2V_DF(df, test_config)
    print("Data Extraction process started ...")
    df = dataExtractor.get_final_df(df,featureNames, test_config)
    print("Results :")
    run(test_config)



    # print("Experiment Parameters")
    # print("classifier params")
    # print(test_config.sampling_strategy)
    # print(test_config.test_ratio)
    # print(test_config.cross_val_k)
    # print(test_config.target)
    # print("n2v params :")
    # print(test_config.n2v_features_num)
    # print(test_config.n2v_return_weight)
    # print(test_config.n2v_walklen)
    # print(test_config.n2v_epochs)
    # print(test_config.n2v_neighbor_weight)
    # print(test_config.pca)
