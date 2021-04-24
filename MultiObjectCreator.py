from DAO import DAO
import itertools
import pandas as pd
from itertools import product
import random
from tqdm import tqdm

class MultiObjectCreator:

    def __init__(self):
        self.dao = DAO()
        self.ref_const_df = self.dao.get_const_ref()
        self.my_df = pd.DataFrame()
        self.max_pos_pairs_ctr = 0
        self.max_neg_pairs_ctr = 20
        self.deleted_models_num = 0
        self.total_pos = 0
        self.total_neg = 0
        self.not_found = 0
        self.other = 0

    def create_pairs_df(self, df, target):

        self.df_objects = df
        all_models_ids = self.ref_const_df['ModelID'].unique()

        print("Creating pairs dataframe ...")
        for model_id in tqdm(all_models_ids):

            positive_pairs_in_model_number = 0
            negative_pairs_in_model_number = 0
            positive_combinations = set()

            consts_ids_in_model = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ConstraintID'].unique()

            for const_id_in_model in consts_ids_in_model:
                objs_in_const_df = self.ref_const_df.loc[self.ref_const_df['ConstraintID'] == const_id_in_model]
                objs_in_const_number = objs_in_const_df.shape[0]

                # ADDING POSITIVE RECORDS
                if objs_in_const_number > 1:
                    object_ids_in_const_set = set(objs_in_const_df['ObjectID'].tolist())

                    for positive_pair in itertools.combinations(sorted(object_ids_in_const_set), 2):
                        positive_combinations.add(positive_pair)  # added
                        positive_pairs_in_model_number = self.add_pair(positive_pair[0],positive_pair[1], 1, model_id, positive_pairs_in_model_number)

            # IF POSITIVE RECORDS ADDED, CREATE ALSO NEGATIVE RECORDS
            if positive_pairs_in_model_number > 0:
                all_obj_ids_in_model = self.df_objects.loc[self.df_objects['ModelID'] == model_id]['ObjectID'].tolist()
                negative_combs_set = self.create_negative_combinations(all_obj_ids_in_model,positive_combinations
                                                             )
                # if cant create equal number of negative records, delete positive pairs
                if len(negative_combs_set) < positive_pairs_in_model_number:
                    self.my_df = self.my_df[self.my_df['ModelID_1'] != model_id]
                    self.deleted_models_num += 1
                    continue
                for negative_pair in negative_combs_set:
                    # STOP WHEN POSITIVE RECORDS NUM = NEGATIVE RECORDS NUM
                    if negative_pairs_in_model_number == self.max_neg_pairs_ctr:
                        break
                    negative_pairs_in_model_number = self.add_pair(negative_pair[0], negative_pair[1], 0, model_id, negative_pairs_in_model_number)

        # self.my_df.to_csv("pairs_un_balanced.csv", index=False)

        self.my_df = self.my_df.drop("ModelID_1",axis=1)
        self.my_df = self.my_df.drop("ModelID_2",axis=1)

        self.print_pairs_creation_stats()

        return self.my_df

    def print_pairs_creation_stats(self):
        print('-' * 50)
        print("pairs creation stats: ")
        print("Input : {} Models". format(len(self.ref_const_df['ModelID'].unique())))
        print("Output : {} Models ". format(len(self.my_df['ModelID'].unique())))
        print("Number of deleted models : {}" .format(self.deleted_models_num))
        print("Number of positive records found in objects dataframe : {}" .format(self.total_pos))
        print("Number of negative records found in objects dataframe : {}" .format(self.total_neg))
        print("Number of times pair wasnt found in objects dataframe : {}" .format(self.not_found))
        print("Number of issues : {}" .format(self.other))





    def subtract_sets(self, x, y):
        final_set = set()
        while y:
            sub = y.pop()
            if sub in x:
                pass
            elif (sub[1], sub[0]) in x:
                pass
            else:
                final_set.add(sub)
        return final_set


    def get_features(self,features):
        final_features = []
        for feature in features:
            final_features.append(feature + str("_1"))
            final_features.append(feature + str("_2"))
        return final_features


    def add_pair(self, obj_id_1, obj_id_2, target_var,model_id,ctr):
        obj_1 = self.df_objects.loc[self.df_objects['ObjectID'] == obj_id_1].reset_index(drop=True)
        obj_2 = self.df_objects.loc[self.df_objects['ObjectID'] == obj_id_2].reset_index(drop=True)
        if (not obj_1.empty) and (not obj_2.empty):
            new_df = obj_1.join(obj_2, lsuffix="_1", rsuffix="_2")
            new_df['PairInConstraint'] = target_var
            new_df['ModelID'] = model_id
            self.my_df = pd.concat([self.my_df, new_df], axis=0)
            if target_var == 1:
                self.total_pos += 1
            if target_var == 0:
                self.total_neg += 1
            ctr += 1
        else:
            self.not_found += 1
        return ctr

    def create_negative_combinations(self, all_obj_ids_in_model, positive_combinations):
        all_combinations = set(itertools.combinations(sorted(all_obj_ids_in_model), 2))
        combinations = set(positive_combinations ^ all_combinations)
        combinations = list(combinations)
        random.shuffle(combinations)
        return combinations