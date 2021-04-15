
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from DAO import DAO
import pandas as pd
import pickle
import numpy as np

class Sampler:


    def __init__(self,df,target):
        self.under_sampler = RandomUnderSampler()
        self.over_sampler = RandomOverSampler()
        self.df = df
        self.new_df = pickle.loads(pickle.dumps(df))
        self.new_df = self.new_df[0:0]
        self.good_models_ctr = 0
        self.bad_models_ctr = 0
        self.models_with_more_pos = 0
        self.models_with_more_neg = 0
        self.target = target

    def sample(self):
        for i in range(1,386):
            filtered_rows = self.df[(self.df['ModelID'] == i)]
            self.model_sample(filtered_rows)

        print('-' * 50)
        print("Sampler stats:")
        print("Added models with balanced target var : " + str(self.good_models_ctr))
        print("Deleted models with un-balanced target var : " + str(self.bad_models_ctr))
        print("Over-sampled models : " + str(self.models_with_more_pos))
        print("Under-sampled models: " + str(self.models_with_more_neg))
        print('-' * 50)
        print()

        return self.new_df


    def model_sample(self,filtered_rows):
        X = filtered_rows.loc[:,  filtered_rows.columns != self.target]
        y = filtered_rows[self.target]
        if len(np.unique(y)) == 2:
            num_pos = filtered_rows[(filtered_rows[self.target] == 1)].shape[0]
            num_neg = filtered_rows[(filtered_rows[self.target] == 0)].shape[0]
            if num_neg > num_pos :
                n_X, n_y = self.under_sampler.fit_resample(X, y)
                n_X[self.target] = n_y
                merged_block = n_X
                self.models_with_more_neg += 1

            else:
                n_X, n_y = self.over_sampler.fit_resample(X, y)
                n_X[self.target] = n_y
                merged_block = n_X
                self.models_with_more_pos += 1
            self.new_df = pd.concat([self.new_df, merged_block], axis=0)
            self.good_models_ctr += 1
        if len(np.unique(y)) == 1:
            self.bad_models_ctr += 1


