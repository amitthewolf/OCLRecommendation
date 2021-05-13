
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from DAO import DAO
import pandas as pd
import pickle
import numpy as np
from Classification.Logger import Logger

class Sampler:


    def __init__(self, df, test_config):
        self.logger = Logger()
        self.dao = DAO()
        self.under_sampler = RandomUnderSampler()
        self.over_sampler = RandomOverSampler()
        self.df = df
        self.new_df = pickle.loads(pickle.dumps(df))
        self.new_df = self.new_df[0:0]
        self.good_models_ctr = 0
        self.bad_models_ctr = 0
        self.models_with_more_pos = 0
        self.models_with_more_neg = 0
        self.models_with_equal_target = 0
        self.target = test_config.target
        self.models_number = test_config.models_number


    def sample(self):
        models_ids = self.df['ModelID'].unique()
        for i in models_ids:
            # filtered_rows = self.df[(self.df['ModelID_1'] == i)]
            filtered_rows = self.df[(self.df['ModelID'] == i)]
            if not filtered_rows.empty:
                self.model_sample(filtered_rows)

        self.print_sampler_stats()
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

            elif num_neg < num_pos:
                n_X, n_y = self.over_sampler.fit_resample(X, y)
                n_X[self.target] = n_y
                merged_block = n_X
                self.models_with_more_pos += 1
            else:
                merged_block = filtered_rows
                self.models_with_equal_target += 1
            self.new_df = pd.concat([self.new_df, merged_block], axis=0)
            self.good_models_ctr += 1
        if len(np.unique(y)) == 1:
            # if np.unique(y)[0] == 1:
            #     X[self.target] = y
            #     merged_block = X
            #     self.new_df = pd.concat([self.new_df, merged_block], axis=0)
            self.bad_models_ctr += 1


    def print_sampler_stats(self):
        print('-' * 50)
        print("Sampler stats : \n")
        print(" Input : {} models".format(len(self.df['ModelID'].unique())))
        print(" Output : {} models composed of : ".format(len(self.new_df['ModelID'].unique())))
        print("     {} Over-sampled models ".format(self.models_with_more_pos))
        print("     {} Under-sampled models ".format(self.models_with_more_neg))
        print("     {} Equal target models ".format(self.models_with_equal_target))
        print(" Etc:")
        print("     {} Models with 1 target value were deleted".format(self.bad_models_ctr))
        print('-' * 50 + " \n  ")

    def LogSamples(self,modelName, XTest, YTest, PredTest, test_config,ObjectIDInOrder, ModelIDInOrder):
        FPCounter = 0
        FNCounter = 0
        for index in range(len(PredTest)):
            if PredTest[index] == 1 and YTest.array[index] == 0 and FPCounter < 3:
                FPCounter += 1
                SampleToLog = XTest.iloc[index]
                Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
                Log_DF['Model'] = modelName
                if test_config.method != 'pairs':
                    ObjectIDRow = ObjectIDInOrder.iloc[index]
                    CurrObjectID = ObjectIDRow.item()
                    ModelRow = self.dao.getModelRowByObjectID(CurrObjectID)
                    ModelList = list(ModelRow[0])
                    Log_DF['ModelID'] = ModelList[0]
                    Log_DF['Path'] = ModelList[1]
                    Log_DF['ConstraintsNum'] = ModelList[2]
                    Log_DF['ObjectsNum'] = ModelList[3]
                else:
                    ModelIDRow = ModelIDInOrder.iloc[index]
                    CurrModelID = ModelIDRow.item()
                    ModelRow = self.dao.getSpecificModel(CurrModelID)
                    ModelList = list(ModelRow)
                    Log_DF['ModelID'] = ModelList[0]
                    Log_DF['Path'] = ModelList[1]
                    Log_DF['ConstraintsNum'] = ModelList[2]
                    Log_DF['ObjectsNum'] = ModelList[3]
                self.logger.LogSamples(Log_DF, 'FP', index=False)
            elif PredTest[index] == 0 and YTest.array[index] == 1 and FNCounter < 3:
                FNCounter += 1
                SampleToLog = XTest.iloc[index]
                Log_DF = pd.DataFrame(data=[SampleToLog], index=['0'], columns=XTest.columns)
                Log_DF['Model'] = modelName
                if test_config.method != 'pairs':
                    ObjectIDRow = ObjectIDInOrder.iloc[index]
                    CurrObjectID = ObjectIDRow.item()
                    ModelRow = self.dao.getModelRowByObjectID(CurrObjectID)
                    ModelList = list(ModelRow[0])
                    Log_DF['ModelID'] = ModelList[0]
                    Log_DF['Path'] = ModelList[1]
                    Log_DF['ConstraintsNum'] = ModelList[2]
                    Log_DF['ObjectsNum'] = ModelList[3]
                else:
                    ModelIDRow = ModelIDInOrder.iloc[index]
                    CurrModelID = ModelIDRow.item()
                    ModelRow = self.dao.getSpecificModel(CurrModelID)
                    ModelList = list(ModelRow)
                    Log_DF['ModelID'] = ModelList[0]
                    Log_DF['Path'] = ModelList[1]
                    Log_DF['ConstraintsNum'] = ModelList[2]
                    Log_DF['ObjectsNum'] = ModelList[3]
                self.logger.LogSamples(Log_DF, 'FN', index=False)
            if FNCounter == 3 and FPCounter == 3:
                return
