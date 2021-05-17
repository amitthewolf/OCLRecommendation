from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import statistics


class PairClassifier:

    instance = None

    def __init__(self, test_config):
        if self.instance != None:
            raise Exception("This class is a singleton!")
        else:
            PairClassifier.instance = self
            self.test_config = test_config

    @staticmethod
    def getInstance(test_config):
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
                results.append(self.predictModel(X_train, X_test, y_train, y_test))

        self.printTestInfo(bal_df,unbal_df)
        self.printResults(results)


    def predictModel(self, X_train, X_test, y_train, y_test):
        models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
        results = {}

        for model in models:
            if model.__class__.__name__ == 'KNeighborsClassifier' and X_train.shape[0] > 4:
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                train_preds = model.predict(X_train)
                results[model.__class__.__name__] = accuracy_score(y_test, test_preds)
            elif model.__class__.__name__ != 'KNeighborsClassifier':
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                train_preds = model.predict(X_train)
                results[model.__class__.__name__] = accuracy_score(y_test, test_preds)
            results["roc_auc_score"] = roc_auc_score(y_test, test_preds)
            results["f1_score"] = f1_score(y_test, test_preds)

        return results

    def printResults(self, results):

        NB = []
        KNN = []
        RF = []
        roc_auc_score = []
        f1 = []

        for result_set in results:
            try:
                NB.append(result_set['GaussianNB'])
                KNN.append(result_set['KNeighborsClassifier'])
                RF.append(result_set['RandomForestClassifier'])
                roc_auc_score.append(result_set['roc_auc_score'])
                f1.append(result_set['f1_score'])
            except Exception as e:
                print(e)

        print(" \n \n \n ")
        print("*" * 50)
        print("Pairs Classification Results : \n ")
        print("*" * 50)

        print("Train on N-1 balanced models and test on 1 un-balanced model avg accuracy: \n")

        print('     GaussianNB :  %0.2f: ' % statistics.mean(NB))
        print('     KNeighborsClassifier :  %0.2f: ' % statistics.mean(KNN))
        print('     RandomForestClassifier :  %0.2f: ' % statistics.mean(RF))
        print('     ROC_AUC Score ' + str(statistics.mean(roc_auc_score)))
        print('     f1-score ' + str(statistics.mean(f1)))

    def printTestInfo(self, bal_df, unbal_df):
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
        print(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))






