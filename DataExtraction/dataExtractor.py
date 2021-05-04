import numpy as np
import pandas as pd
from configparser import ConfigParser
from DAO import DAO
from DataExtraction.node2vec import node2vec as Node2Vec
from Classification.Sampler import Sampler
from DataExtraction.MultiObjectCreator import MultiObjectCreator

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
        self.creator = MultiObjectCreator()

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

    def get_final_df(self, df, features, test_config):

        # DataExtraction.append("ObjectID")
        # Set current test properties
        self.curr_test_config = test_config
        self.final_features = features

        # Add DataExtraction + label
        df = self.add_N2V_features(df, test_config)
        df = self.add_inherit_feature(df)
        df = self.add_objects_number_in_model_feature(df)
        df = self.add_graphlets_features(df)
        df = self.add_target_variable(df,test_config.target)


        if test_config.method == 'pairs':
            pairs_balanced_df, pairs_un_balanced_df = self.handle_pairs_dataframes(df, test_config)
            return pairs_balanced_df, pairs_un_balanced_df

        if test_config.method == 'ones':
            samp = Sampler(df, test_config)
            df = samp.sample()

        if test_config.method == 'operator':
            df = df.loc[df['ConstraintsNum'] > 0 ]

        self.final_features.append(test_config.target)
        df = df[self.final_features]
        df = df.dropna()

        return df


    def drop_irrelevant_features_and_na(self,df,target):
        feat = self.final_features
        df = df[feat]
        df = df.dropna()
        df = df.drop_duplicates()

        return df

    def handle_pairs_dataframes(self, df, test_config):
        if test_config.pairs_creation_flag == 'True':
            pairs_un_balanced_df = self.creator.create_pairs_df(df, test_config.target)
            pairs_un_balanced_df.to_csv(self.paths['UNBALANCED_PAIRS'], index=False )
        else:
            pairs_un_balanced_df = pd.read_csv(self.paths['UNBALANCED_PAIRS'])
        samp = Sampler(pairs_un_balanced_df, test_config)
        pairs_balanced_df = samp.sample()

        self.final_features = self.creator.get_features(self.final_features)
        self.final_features.append("ModelID")
        self.final_features.append(test_config.target)
        pairs_balanced_df = self.drop_irrelevant_features_and_na(pairs_balanced_df, test_config.target)
        pairs_un_balanced_df = self.drop_irrelevant_features_and_na(pairs_un_balanced_df, test_config.target)

        pairs_balanced_df.to_csv("pairs_balanced.csv", index=False)
        return pairs_balanced_df, pairs_un_balanced_df


