from configparser import ConfigParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from DataExtraction.dataExtractor import dataExtractor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import math
import numpy as np
from Classification.TestConfig import TestConfig
from DAO import DAO



class Tuner:

    def __init__(self):
        self.dao = DAO()
        config = ConfigParser()
        self.dataExtractor = dataExtractor()
        config.read('conf.ini')
        fixed_section = config['fixed_params']
        random_param_sampling = fixed_section['random']
        test_method = fixed_section['test_method']
        models_number = self.dao.get_models_number()
        graphlets_flag = fixed_section['graphlets_flag']

        if random_param_sampling == 'True':
            self.test_config = TestConfig(graphlets_flag, models_number, test_method)
        else:
            self.test_config = TestConfig(graphlets_flag, models_number, test_method, random=False)

    def get_best_params(self, X_train, X_test, y_train, y_test, score):
        models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]

        print("Tuning Hyper-Parameters ... \n ")

        for model in models:

            x = model.__class__.__name__
            if x == 'RandomForestClassifier':
                y = list(np.arange(50, 100, 200))
                param_grid_rf = {
                    'n_estimators': y,
                    'max_features': ['auto', 'sqrt', 'log2']}
                CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=5, scoring=score)
                CV_rfc.fit(X_train, y_train)
                clf = CV_rfc

            elif x == 'GaussianNB':
                params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
                gs_NB = GridSearchCV(estimator=model, param_grid=params_NB, cv=5, scoring=score)
                gs_NB.fit(X_train, y_train)
                clf = gs_NB

            elif x == 'KNeighborsClassifier':
                sqrt_samples = math.sqrt(X_train.shape[0])
                y = np.arange(1, sqrt_samples, 2)
                arr = list(np.nan_to_num(y, copy=False).astype(np.int))
                # 'metrics': ['minkowski', 'euclidean', 'manhattan'],
                params = {'weights': ['uniform', 'distance'], 'n_neighbors': arr}
                knn_clf = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring=score)
                knn_clf.fit(X_train, y_train)
                clf = knn_clf

            print("Model : {}".format(x))
            print("Best Params : {}".format(clf.best_params_))
            other_scores = cross_val_score(clf, X_train, y_train, cv=self.test_config.cross_val_k)
            print("%0.2f Cross-Validation average accuracy with a standard deviation of %0.2f" % (
            other_scores.mean(), other_scores.std()))
            print('-' * 50)


    def getOnesData(self):
            featureNames = self.test_config.classifier_section['featureNames'].split(',')
            df = self.dao.getObjects()
            self.test_config.update_iteration_params(0)
            self.df = self.dataExtractor.get_final_df(df, featureNames, self.test_config)
            return self.df


    def tune(self,df):
        X = df.loc[:, df.columns != self.test_config.target]
        y = df[self.test_config.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_config.test_ratio, random_state=0)
        self.get_best_params(X_train, X_test, y_train, y_test,'accuracy') # (f1\accuracy)


tuner = Tuner()
df = tuner.getOnesData()
tuner.tune(df)



