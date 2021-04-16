from DAO import DAO
import numpy as np
import pandas as pd
from configparser import ConfigParser
from DAO import DAO
from node2vec import node2vec as Node2Vec
from Sampler import Sampler

class dataExtractor:

    def __init__(self):
        self.config = ConfigParser()
        self.config.read('conf.ini')
        self.dao = DAO()
        self.N2V_df = {}
        self.n2v_features = {}
        self.final_features = []
        self.curr_test_config = None

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
        if n2v_section['n2v_flag'] == 'True':
            print("N2V process started ...")
            n2v = Node2Vec(test_config.n2v_features_num,
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
            graphlets = pd.read_csv("final_graphlet_features_model_is_file.csv")
            merged_df = pd.concat((df, graphlets), axis=1)
            grap_feat = ["O" + str(i) for i in range(0, 73)]
            self.final_features += grap_feat
            self.check_oo_rn(merged_df)
            return merged_df
        return df

    def check_oo_rn(self,df):
        good_ids = []
        bad_ids = []
        for indrx,row in df.iterrows():
            if row['RelationNum'] != row['O0']:
                bad_ids.append(row['ObjectID'])
            if row['RelationNum'] == row['O0']:
                good_ids.append(row['ObjectID'])

        print("#" * 50)
        print("Graphlet bad rows: " + str(len(bad_ids)))
        print("Graphlet good rows: " + str(len(good_ids)))
        print("Total objects: " + str(df.shape[0]))
        print()
        print("#" * 50)


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

        # Set current test properties
        self.curr_test_config = test_config
        self.final_features = features

        # Add features + label
        df = self.add_N2V_features(df, test_config)
        df = self.add_inherit_feature(df)
        df = self.add_objects_number_in_model_feature(df)
        df = self.add_graphlets_features(df)
        df = self.add_target_variable(df,test_config.target)

        #Sample
        samp = Sampler(df, test_config.target)
        df = samp.sample()

        self.final_features.append(test_config.target)
        df = df[self.final_features]
        df = df.dropna()


        return df