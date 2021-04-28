## Importing required libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from configparser import ConfigParser
from DAO import DAO
from Classification.TestConfig import TestConfig
from DataExtraction.dataExtractor import dataExtractor
from sklearn.model_selection import cross_val_score
from Classification.Logger import Logger
import statistics

def classify(X_train, X_test, y_train, y_test,feature_names):
    models = [GaussianNB(), KNeighborsClassifier(),RandomForestClassifier()]


    for model in models:
        try:
            print(model.__class__.__name__ + " : ")
            # "Regular" classifiers
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            train_preds = model.predict(X_train)
            # print(" Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
            # print(" Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))
            print()
            if test_config.method != 'operator':
                scores = cross_val_score(model, X_train, y_train, cv=test_config.cross_val_k)
                # print("Cross-Validation result for k = {} : ".format(test_config.cross_val_k))
                # print('Scores :  {} '.format(scores))
                print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
                print('-' * 50)
                # LogSamples(model.__class__.__name__,feature_names, X_test, y_test, test_preds)
                # LogResult(model.__class__.__name__,y_test, test_preds,y_train, train_preds,scores)
            else:
                scores = 0
                YLen = len(y_test[0])
                NumofAnswers = len(y_test)
                for ans in range(len(y_test)):
                    Prediction = test_preds[ans]
                    Actual = y_test[ans]
                    currScore = 0
                    tempYLen = YLen
                    for index in range(YLen):
                        if Prediction[index]==Actual[index]:
                            if Prediction[index] == 0:
                                tempYLen -= 1
                            else:
                                currScore += 1
                    if tempYLen != 0:
                        scores += currScore/tempYLen
                    else:
                        NumofAnswers -= 1
                scores = scores/NumofAnswers
                print(scores)


        except Exception as e:
            print(e)
            print(" Invalid Y Dimentions")


def predict_pairs(X_train, X_test, y_train, y_test):
    models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    results = {}

    for model in models:
        # print(model.__class__.__name__ + " : ")
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        results[model.__class__.__name__ ] = accuracy_score(y_test, test_preds)

    return results


def LogResult(modelName, YTest,PredTest, YTrain, PredTrain,scores):
    if test_config.n2v_flag == 'True':
        data = {'Features': ','.join(featureNames[:-1]),
                'Timestamp': datetime.now(),
                'Sampling': test_config.sampling_strategy,
                'Test_ratio': test_config.test_ratio,
                'Cross_Val_K': test_config.cross_val_k,
                'Graphlets': fixed_section['graphlets'],
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
                'Train Score': accuracy_score(YTrain, PredTrain),
                'Test Score': accuracy_score(YTest, PredTrain),
                'Mean': scores.mean(),
                'Std': scores.std(),
                }

        Log_DF = pd.DataFrame(data, columns=['Features',
                                             'Timestamp',
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
                'Graphlets': fixed_section['graphlets'],
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
                                             'Target',
                                             'Model',
                                             'Train Score',
                                             'Test Score',
                                             'Mean',
                                             'Std'], index=[0])
        print(Log_DF)
    log = Logger()
    log.append_df_to_excel(Log_DF, header=None, index=False)

def LogSamples(modelName,feature_names, XTest, YTest, PredTest):
    FPCounter = 0
    FNCounter = 0
    for index in range(len(PredTest)):
        if PredTest[index]==1 and YTest.array[index]==0 and FPCounter<3:
            FPCounter += 1
            SampleToLog = XTest.iloc[index]
            log = Logger()
            Log_DF = pd.DataFrame(data=[SampleToLog],index=['0'],columns=feature_names)
            log.LogSamples(Log_DF,'FP', index=False)
        elif PredTest[index] == 0 and YTest.array[index] == 1 and FNCounter<3:
            FNCounter += 1
            SampleToLog = XTest.iloc[index]
            log = Logger()
            #Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=feature_names)
            #log.LogSamples(Log_DF, 'FN', index=False)
        if FNCounter==3 and FPCounter == 3:
            return

def getOperatorY(ObjectIDs):
    return dao.GetConstraintRandomSample(ObjectIDs)

def run(df,test_config):
    X = df.loc[:, df.columns != test_config.target ]
    X = X.loc[:, X.columns != "ObjectID"]


    print("Test Statistics :")
    print("-" * 50)
    print("Dataframe columns :    " + str(list(X.columns)))
    print("-" * 50)


    if test_config.method == 'operator':
        y = getOperatorY(df["ObjectID"])
    else:
        y = df[test_config.target]
        print("Number of positive records : " + str(df[df[test_config.target] == 1].shape[0]))
        print("Number of negative records : " + str(df[df[test_config.target] == 0].shape[0]))
        print("-" * 25 + "Mutual Information" + "-" * 25)
        feature_names = X.columns
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_config.test_ratio, random_state=0)

    # print("-" * 25 + " Sampling Data " + "-" * 25)
    # print("Number of Rows in Data after sampling is: " + str(X.shape[0]))

    print("-" * 25 + " Data stats " + "-" * 25)
    print("Number of Rows in Train Set is : " + str(X_train.shape[0]))
    print("Number of Rows in Test Set is : " + str(X_test.shape[0]))
    print()

    print(datetime.now())
    # str1 = ''.join(featuresNames)
    # print("DataExtraction: "+ ''.join(featuresNames))
    print("DataExtraction: " + str(list(df.columns)))

    # n2v_feat = "Features: " + n2v_section['n2v_features_num'] + ", Attributes: " + n2v_section['n2v_use_attributes'] + \
    #            ", Inheritance: " + n2v_section['n2v_use_inheritance'] + ", Return weight: " + \
    #            n2v_section['n2v_return_weight'] + ", Walklen: " + n2v_section['n2v_walklen'] + ", Epcochs:" + \
    #            n2v_section['n2v_epochs'] + ", Neighbour weight: " + n2v_section['n2v_neighbor_weight'] + \
    #            ", Use PCA: " + n2v_section['use_pca'] + ", PCA num: " + n2v_section['pca_num']
    #
    # if n2v_section['n2v_flag'] == 'True':
    #     print('Node2Vec Features:')
    #     print(n2v_feat)

    #("sampling strategy: " + test_config.sampling_strategy)
    print("-" * 25 + " Results " + "-" * 25)
    classify(X_train, X_test, y_train, y_test,feature_names)

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

        results.append(predict_pairs(X_train,X_test,y_train,y_test))

    NB = []
    KNN = []
    RF = []
    for result_set in results:
        NB.append(result_set['GaussianNB'])
        KNN.append(result_set['KNeighborsClassifier'])
        RF.append(result_set['RandomForestClassifier'])

    print('GaussianNB ' + str(statistics.mean(NB)))
    print('KNeighborsClassifier ' + str(statistics.mean(KNN)))
    print('RandomForestClassifier ' + str(statistics.mean(RF)))

    print("bal DataExtraction: {} ".format(list(b_df.columns)))
    print("unbal DataExtraction: {} ".format(list(ub_df.columns)))





dao = DAO()
config = ConfigParser()
dataExtractor = dataExtractor()


#get configurations
config.read('conf.ini')
fixed_section = config['fixed_params']
iterations = int(fixed_section['iterations'])
random_param_sampling = fixed_section['random']
test_method = fixed_section['test_method']

models_number = dao.get_models_number()

graphlets_flag = fixed_section['graphlets_flag']

if random_param_sampling == 'True':
    test_config = TestConfig(graphlets_flag, models_number,test_method)
else:
    test_config = TestConfig(graphlets_flag,models_number,test_method,random=False)


for i in range(iterations):
    featureNames = test_config.classifier_section['featureNames'].split(',')
    df = dao.getObjects()
    # print('*' * 50)
    # print("{} Experiment ".format(i + 1))
    # print('*' * 50)
    test_config.update_iteration_params(i)
    if test_method != 'pairs':
        df = dataExtractor.get_final_df(df,featureNames, test_config)
        run(df,test_config)
    elif test_method == 'pairs':
        b_df, ub_df = dataExtractor.get_final_df(df,featureNames, test_config)
        prepare_pairs_test_train(b_df,ub_df)






