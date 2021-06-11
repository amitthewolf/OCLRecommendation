## Importing required libraries

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from configparser import ConfigParser
from DAO import DAO
from Classification.TestConfig import TestConfig
from DataExtraction.dataExtractor import dataExtractor
from sklearn.model_selection import cross_val_score
from Classification.Logger import Logger
from Classification.GroupClassifier import PairClassifier
from Classification.OperatorClassifier import OperatorClassifier
ObjectIDInOrder = None
ModelIDInOrder = None


def classify(X_train, X_test, y_train, y_test):
    models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]

    for model in models:
        print(model.__class__.__name__ + " : ")
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        print("\t Accuracy on Test Set " + str(accuracy_score(y_test, test_preds)))
        print("\t Accuracy on Train Set " + str(accuracy_score(y_train, train_preds)))
        scores = cross_val_score(model, X_train, y_train, cv=test_config.cross_val_k)
        print("\t %0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print("\t Roc-Auc score : {} ".format(round(roc_auc_score(y_test, test_preds),2)))
        print("\t F1-score : {} ".format(round(f1_score(y_test, test_preds),2)))
        print('-' * 50)

        #sampler.LogSamples(model.__class__.__name__, X_test, y_test, test_preds, test_config, ObjectIDInOrder, ModelIDInOrder)
        #logger.LogResult(model.__class__.__name__, y_test, test_preds, y_train, train_preds, scores,test_config, featureNames, iterations, fixed_section)



def getOperatorY(ObjectIDs):
    return dao.GetConstraintRandomSample(ObjectIDs)

def predict_ones(df, test_config):
    X = df.loc[:, df.columns != test_config.target]
    X = X.loc[:, X.columns != "ObjectID"]
    ObjectIDInOrder = df["ObjectID"]

    print("*" * 50)
    print("Test Information : ")
    print("*" * 50)

    print("-" * 50)
    print("Dataframe cols :    " + str(list(X.columns)))
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
    print("\n \n \n ")

    print("*" * 50)
    print("Test Results : ")
    print("*" * 50)
    classify(X_train, X_test, y_train, y_test)




dao = DAO()
config = ConfigParser()
dataExtractor = dataExtractor()
logger = Logger()
op_clf = OperatorClassifier()
#sampler = Sampler(None, None)

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
        ObjectIDInOrder = df['ObjectID']
        predict_ones(df, test_config)
    elif test_method == 'pairs':
        b_df, ub_df, ModelIDInOrder = dataExtractor.get_final_df(df, featureNames, test_config)
        pairs_clf = PairClassifier.get_instance(test_config)
        pairs_clf.predict_groups(b_df, ub_df)
