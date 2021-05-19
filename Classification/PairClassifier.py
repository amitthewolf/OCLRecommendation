import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import statistics
import json
import pandas as pd


class PairClassifier:

    instance = None

    def __init__(self, test_config):
        if self.instance != None:
            raise Exception("This class is a singleton!")
        else:
            PairClassifier.instance = self
            self.test_config = test_config
            self.roc_scores = {}
            self.roc_scores['GaussianNB'] = []
            self.roc_scores['KNeighborsClassifier'] = []
            self.roc_scores['RandomForestClassifier'] = []

    @staticmethod
    def get_instance(test_config):
        if PairClassifier.instance == None:
            PairClassifier(test_config)
        return PairClassifier.instance

    def predict(self, bal_df, unbal_df):
        results = []
        models_ids = bal_df['ModelID'].unique()

        for model_id in models_ids:

            # filter all models except one
            bal_df_models = bal_df.loc[bal_df['ModelID'] != model_id]
            bal_df_models = bal_df_models.drop("ModelID", axis=1)

            # filter the one model
            unbal_df_model = unbal_df.loc[unbal_df['ModelID'] == model_id]
            unbal_df_model = unbal_df_model.drop("ModelID", axis=1)

            X_train = bal_df_models.loc[:, bal_df_models.columns != self.test_config.target]
            y_train = bal_df_models[self.test_config.target]

            X_test = unbal_df_model.loc[:, unbal_df_model.columns != self.test_config.target]
            y_test = unbal_df_model[self.test_config.target]

            if bal_df_models.shape[0] > 0 and unbal_df_model.shape[0] > 0 :
                results.append(self.predict_model(X_train, X_test, y_train, y_test))


        self.print_test_info(bal_df, unbal_df)
        self.print_test_results(results)

    def predict_model(self, X_train, X_test, y_train, y_test):
        models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
        results = {}

        for model in models:
            model_name = model.__class__.__name__
            if  model_name == 'KNeighborsClassifier' and X_train.shape[0] > 4:
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                results[model_name] = accuracy_score(y_test, test_preds)
                self.roc_scores[model_name].append((y_test.values, test_preds))
            elif model_name != 'KNeighborsClassifier':
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                results[model_name] = accuracy_score(y_test, test_preds)
                self.roc_scores[model_name].append((y_test.values, test_preds))
        return results

    def get_classifiers_predictions(self):
        cols = ['test','preds']
        NB_y_preds = pd.DataFrame(columns=cols)
        KNN_y_preds = pd.DataFrame(columns=cols)
        RF_y_preds = pd.DataFrame(columns=cols)

        for key,tuples in self.roc_scores.items():
            for pair in tuples:
                test = pair[0]
                preds = pair[1]
                d = {cols[0]: test, cols[1]: preds}
                df = pd.DataFrame(data=d)
                if key == 'GaussianNB':
                    NB_y_preds = NB_y_preds.append(df)
                if key == 'KNeighborsClassifier':
                    KNN_y_preds = KNN_y_preds.append(df)
                if key == 'RandomForestClassifier':
                    RF_y_preds = RF_y_preds.append(df)


        return NB_y_preds, KNN_y_preds, RF_y_preds

    def print_test_info(self, bal_df, unbal_df):
        X = bal_df.loc[:, bal_df.columns != self.test_config.target]
        y = bal_df[self.test_config.target]

        print("*" * 50)
        print("Test Information : ")
        print("*" * 50)

        print("-" * 50)
        print("Balanced Dataframe cols :    " + str(list(bal_df.columns)))
        print("Un-Balanced Dataframe cols :    " + str(list(unbal_df.columns)))
        print("-" * 50)

        print("Number of positive records in balanced train set : " + str(bal_df[bal_df[self.test_config.target] == 1].shape[0]))
        print("Number of negative records in balanced train set : " + str(bal_df[bal_df[self.test_config.target] == 0].shape[0]))

        print("Number of positive records in un-balanced test set : " + str(unbal_df[unbal_df[self.test_config.target] == 1].shape[0]))
        print("Number of negative records in un-balanced test set : " + str(unbal_df[unbal_df[self.test_config.target] == 0].shape[0]))

        print("-" * 25 + "Mutual Information" + "-" * 25)
        res = mutual_info_classif(X, y)
        mi_dict = dict(zip(X.columns, res))
        mi_dict_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

        for mi_feature in mi_dict_sorted[:10]:
            print(mi_feature)

        with open('../Outputs/MutualInfo.txt', 'w') as csv_file:
            for element in mi_dict_sorted:
                csv_file.write(element[0] + str("     ") + (str(element[1])))
                csv_file.write("\n")

    def print_test_results(self, results):

        NB = []
        KNN = []
        RF = []

        NB_y_preds, KNN_y_preds, RF_y_preds = self.get_classifiers_predictions()

        for result_set in results:
            try:
                NB.append(result_set['GaussianNB'])
                KNN.append(result_set['KNeighborsClassifier'])
                RF.append(result_set['RandomForestClassifier'])
            except Exception as e:
                print(e)

        print("\n")
        print("*" * 50)
        print("Pairs Classification Results :")
        print("*" * 50)

        print('\t GaussianNB ')
        print('\t\t accuracy %0.2f: ' % statistics.mean(NB))
        print("\t\t Roc {} ".format(roc_auc_score(NB_y_preds['test'].tolist(), NB_y_preds['preds'].tolist())))
        print("\t\t f1 {} ".format(f1_score(NB_y_preds['test'].tolist(), NB_y_preds['preds'].tolist())))
        print()
        print('\t KNeighborsClassifier :')
        print('\t\t accuracy %0.2f: ' % statistics.mean(KNN))
        print("\t\t Roc {} ".format(round(roc_auc_score(KNN_y_preds['test'].tolist(), KNN_y_preds['preds'].tolist()),2)))
        print("\t\t f1 {} ".format(round(f1_score(KNN_y_preds['test'].tolist(), KNN_y_preds['preds'].tolist()),2)))
        print()
        print('\t RandomForestClassifier :')
        print('\t\t accuracy %0.2f: ' % statistics.mean(RF))
        print("\t\t Roc {} ".format(round(roc_auc_score(RF_y_preds['test'].tolist(), RF_y_preds['preds'].tolist()),2)))
        print("\t\t f1 {} ".format(round(f1_score(RF_y_preds['test'].tolist(), RF_y_preds['preds'].tolist()),2)))






