from DAO import DAO
import numpy as np
import pandas as pd
from configparser import ConfigParser

class dataExtractor:

    def __init__(self):
        self.config = ConfigParser()
        self.config.read('conf.ini')

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


    def get_final_df(self,df):
        features_to_retain = self.config['classifier']['featureNames'].split(',')
        TargetVariable = self.config['classifier']['Target']
        features_to_retain.append(TargetVariable)
        df['ContainsConstraints'] = df.apply(lambda x: self.CheckifConstraint(x['ConstraintsNum']), axis=1)
        df['inherits'] = df.apply(lambda x: self.inherits_column(x['inheriting_from']), axis=1)
        df = df[features_to_retain]
        return df
        # df['Referenced'] = df['ReferencedInConstraint'].fillna(0)