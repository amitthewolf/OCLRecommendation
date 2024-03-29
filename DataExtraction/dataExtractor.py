import pickle

import numpy as np
import pandas as pd
from configparser import ConfigParser

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from DAO import DAO
from DataExtraction.node2vec import node2vec as Node2Vec
from Classification.Sampler import Sampler
from DataExtraction.GroupCreator import GroupCreator
import random

class dataExtractor:

    def __init__(self):
        self.config = ConfigParser()
        self.config.read('conf.ini')
        self.paths = self.config['paths']
        self.dao = DAO()
        self.N2V_df = {}
        self.n2v_features = {}
        self.final_features = []
        self.curr_test_config = None
        self.creator = GroupCreator()

    def CheckifConstraint(self,genre):
        if genre == 0:
            return 0
        else:
            return 1


    def inherits_column(self,value):
        if not value or value == 0 or value == "":
            return 0
        else:
            return 1


    def CheckifReferenced(self,genre):
        if not genre or genre == 'nan' or genre == 0:
            return 0
        else:
            return 1

    def add_N2V_features(self, df, test_config):
        n2v_section = test_config.n2v_section
        if test_config.n2v_flag == 'True':
            print("N2V process started ...")
            n2v = Node2Vec(df,
                           test_config.n2v_features_num,
                           n2v_section['n2v_use_attributes'],
                           n2v_section['n2v_use_inheritance'],
                           test_config.n2v_return_weight,
                           test_config.n2v_walklen,
                           test_config.n2v_epochs ,
                           test_config.n2v_neighbor_weight,
                           n2v_section['use_pca'], test_config.pca)
            df = n2v.run()
            self.N2V_df = df
            features_num = test_config.n2v_features_num
            if n2v_section['use_pca'] == 'True':
                features_num = test_config.pca
            self.n2v_features = ['N2V_' + str(i) for i in range(1, features_num + 1)]
            self.final_features += self.n2v_features
            return self.N2V_df
        return df

    def add_graphlets_features(self,df):
        if self.curr_test_config.graphlet_flag == 'True':
            graphlets = pd.read_csv(self.paths['GRAPHLETS'])
            merged_df = pd.concat((df, graphlets), axis=1)
            grap_feat = ["O" + str(i) for i in range(0, 73)]
            self.final_features += grap_feat
            #self.check_oo_rn(merged_df)
            return merged_df
        return df


    def add_target_variable(self,df,target):
        if target == 'InConstraint':
            df = self.add_object_in_constraint_label(df)
        if target == 'ContainsConstraints':
            df['ContainsConstraints'] = df.apply(lambda x: self.CheckifConstraint(x['ConstraintsNum']), axis=1)
        return df


    def check_if_object_in_constraint(self,object_id ,const_ref_ids):
        if object_id in const_ref_ids:
            return 1
        else:
            return 0

    def add_objects_number_in_model_feature(self,df):
        models_df = self.dao.get_num_of_objects_in_model()
        df['ObjectsNum'] = df.apply(lambda x: self.check_objects_num_in_model(x['ModelID'],models_df,x['ObjectID']), axis=1)
        return df

    def check_objects_num_in_model(self,model_id,models_df,oid):
        l = models_df[models_df['ModelID']==model_id]['ObjectsNum'].values
        if len(l) == 0:
            return 2
        return l[0]


    def add_object_in_constraint_label(self,df):
        const_ref_ids = self.dao.get_const_ref_table_ids()
        df = df.assign(InConstraint = np.nan)
        df['InConstraint'] = df.apply(lambda x: self.check_if_object_in_constraint(x['ObjectID'],const_ref_ids), axis=1)
        return df


    def add_inherit_feature(self,df):
        df['inherits'] = df.apply(lambda x: self.inherits_column(x['inheriting_from']), axis=1)
        return df




        return train_ones_df,pairs_df

    def get_final_df(self, df, features, test_config):


        if test_config.method != 'pairs':
            features.append("ObjectID")
        # Set current test properties
        self.curr_test_config = test_config
        self.final_features = features

        # Add  features + label
        df = self.add_N2V_features(df, test_config)
        df = self.add_inherit_feature(df)
        df = self.add_objects_number_in_model_feature(df)
        df = self.add_graphlets_features(df)
        df = self.add_target_variable(df,test_config.target)



        if test_config.method == 'pairs':
            pairs_balanced_df, pairs_un_balanced_df,ModelIDInOrder = self.create_groups_dataframe(df, test_config)
            return pairs_balanced_df, pairs_un_balanced_df,ModelIDInOrder

        if test_config.method == 'ones':
            samp = Sampler(df, test_config)
            df = samp.sample()

        if test_config.method == 'operator':
            df = df.loc[df['ConstraintsNum'] > 0 ]

        self.final_features.append(test_config.target)
        df = df[self.final_features]
        df = df.dropna()

        return df


    def create_groups_dataframe(self, df, test_config):

        if self.config['pairs']['add_ones_results_as_feature'] == 'True':
            df['ContainsConstraints'] = df.apply(lambda x: self.CheckifConstraint(x['ConstraintsNum']), axis=1)
            df = self.add_ones_results_as_feature(df)

        if test_config.pairs_creation_flag == 'True':
            pairs_un_balanced_df = self.creator.create_groups_df(df, test_config.target)
            pairs_un_balanced_df.to_csv(self.paths['UNBALANCED_PAIRS'], index=False )
        else:
            pairs_un_balanced_df = pd.read_csv(self.paths['UNBALANCED_PAIRS'])



        self.final_features = self.creator.get_features(self.final_features)
        self.final_features.append("ModelID")
        self.final_features.append(test_config.target)

        ModelIDInOrder = pairs_un_balanced_df['ModelID']

        pairs_un_balanced_df = self.drop_irrelevant_features_and_na(pairs_un_balanced_df,test_config.target)


        samp = Sampler(pairs_un_balanced_df, test_config)
        pairs_balanced_df = samp.sample()

        pairs_balanced_df.to_csv(self.paths['BALANCED_PAIRS'], index=False)

        return pairs_balanced_df, pairs_un_balanced_df,ModelIDInOrder


    def add_ones_results_as_feature(self, df):


        print("Adding Ones Classification Results as Group Classification Features ... ")

        train_ones_df, pairs_df = self.split_df(df)

        ones_target = 'ContainsConstraints'
        features = self.final_features
        features.append(ones_target)

        train_ones_df = train_ones_df[features]
        X_train = train_ones_df.loc[:, train_ones_df.columns != ones_target]
        y_train = train_ones_df[ones_target]

        pairs_df_ids = pairs_df[['ObjectID', 'ModelID']]
        pairs_df = pairs_df[features]
        X_test = pairs_df.loc[:, pairs_df.columns != ones_target]
        features.remove("ContainsConstraints")

        pairs_df_with_ones_feature = self.predict_and_concat(X_train, y_train, X_test, pairs_df_ids)

        return pairs_df_with_ones_feature



    def split_df(self, df):
        df = df.dropna()
        df_copy = pickle.loads(pickle.dumps(df))
        train_ones_df = df_copy[0:0]
        pairs_df = df_copy[0:0]

        models_ids = list(self.dao.get_models_ids())
        ratio = 0.5
        random.shuffle(models_ids)

        for model_id in models_ids:
            model_rows = df.loc[df['ModelID'] == model_id]

            if (train_ones_df.shape[0] / df.shape[0]) < ratio:
                train_ones_df = pd.concat([train_ones_df, model_rows], axis=0)
            else:
                pairs_df = pd.concat([pairs_df, model_rows], axis=0)

        return train_ones_df, pairs_df


    def predict_and_concat(self, X_train, y_train, X_test, pairs_df_ids):

        models = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]

        df = pd.concat([X_test, pairs_df_ids], axis=1)

        for model in models:
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            df[model.__class__.__name__] = test_preds

        return df




    def drop_irrelevant_features_and_na(self,df, target):
        feat = self.final_features
        if target and target not in feat : feat.append(target)
        df = df[feat]
        df = df.dropna()
        df = df.drop_duplicates()
        return df
