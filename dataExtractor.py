from DAO import DAO
import numpy as np
import pandas as pd
from configparser import ConfigParser
from DAO import DAO
from node2vec import node2vec as Node2Vec

class dataExtractor:

    def __init__(self):
        self.config = ConfigParser()
        self.config.read('conf.ini')
        self.dao = DAO()

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


    def createBalancedData(self,df):
        NoCons_indices = df[df.ContainsConstraints == 0].index
        Cons_indices = df[df.ContainsConstraints == 1].index
        random_NoCons = np.random.choice(NoCons_indices, 3005, replace=False)
        RandomNoCons_sample = df.loc[random_NoCons]
        Constraints_sample = df.loc[Cons_indices]
        return pd.concat([RandomNoCons_sample, Constraints_sample], ignore_index=True)
    #
    # def createBalancedDataRef():
    #     NoCons_indices = df[df.Referenced == 0].index
    #     Cons_indices = df[df.Referenced == 1].index
    #     # random_NoCons = np.random.choice(NoCons_indices, 3572, replace=False)
    #     random_NoCons = np.random.choice(NoCons_indices, 10000, replace=False)
    #     RandomNoCons_sample = df.loc[random_NoCons]
    #     Constraints_sample = df.loc[Cons_indices]
    #     return pd.concat([RandomNoCons_sample, Constraints_sample], ignore_index=True)


    def get_final_df(self,df, features, target,n2v_section):

        # ADD_N2V_FEATURES
        if n2v_section['n2v_flag'] == 'True':
            n2v = Node2Vec(n2v_section['n2v_features_num'])
            df = n2v.run()
            features_num = int(n2v_section['n2v_features_num'])
            n2v_features = ['N2V_' + str(i) for i in range(1, features_num + 1)]
            features = features + n2v_features

        #ADD_LABEL
        df = self.add_object_in_constraint_label(df)

        #ADD_MORE_RELEVANT_FEATURES
        df['ContainsConstraints'] = df.apply(lambda x: self.CheckifConstraint(x['ConstraintsNum']), axis=1)
        df['inherits'] = df.apply(lambda x: self.inherits_column(x['inheriting_from']), axis=1)


        features += target
        df = df[features]
        df.dropna(inplace=True)
        return df
        # df['Referenced'] = df['ReferencedInConstraint'].fillna(0)

    def check_if_object_in_constraint(self,object_id ,const_ref_ids):
        if object_id in const_ref_ids:
            return 1
        else:
            return 0

    def add_object_in_constraint_label(self,df):
        const_ref_ids = self.dao.get_const_ref_table_ids()
        df = df.assign(InConstraint = np.nan)
        df['InConstraint'] = df.apply(lambda x: self.check_if_object_in_constraint(x['ObjectID'],const_ref_ids), axis=1)
        df.to_csv('bbbb.csv')
        return df