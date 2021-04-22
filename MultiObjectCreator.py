from DAO import DAO
import itertools
import pandas as pd
from itertools import product
import random


class MultiObjectCreator:

    def __init__(self):
        self.dao = DAO()
        self.ref_const_df = self.dao.get_const_ref()
        self.my_df = pd.DataFrame()
        self.max_pos_pairs_ctr = 0
        self.max_neg_pairs_ctr = 300

    def create_pairs_df(self, df, target):

        self.df_objects = df
        all_models_ids = self.df_objects['ModelID'].unique()

        for model_id in all_models_ids:
            print(model_id)
            positive_combinations = set()

            all_obj_ids_in_model = self.df_objects.loc[self.df_objects['ModelID'] == model_id]['ObjectID'].tolist()
            obj_ids_in_model_with_consts = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ObjectID'].tolist()
            obj_ids_in_model_with_no_const = list(set(all_obj_ids_in_model) ^ set(obj_ids_in_model_with_consts))
            consts_ids_in_model = self.ref_const_df.loc[self.ref_const_df['ModelID'] == model_id]['ConstraintID'].unique()

            positive_pairs_ctr = 0
            negative_pairs_ctr = 0
            for const_id_in_model in consts_ids_in_model:
                objs_in_const_df = self.ref_const_df.loc[self.ref_const_df['ConstraintID'] == const_id_in_model]
                objs_in_const_number = objs_in_const_df.shape[0]

                # ADDING POSITIVE RECORDS
                if objs_in_const_number > 1:
                    objects_ids_in_constraint = objs_in_const_df['ObjectID'].tolist()
                    object_ids_in_const_set = set(objects_ids_in_constraint)
                    for positive_pair in itertools.combinations(sorted(object_ids_in_const_set), 2):
                        positive_combinations.add(positive_pair)  # added
                        obj_1 = self.df_objects.loc[self.df_objects['ObjectID'] == positive_pair[0]].reset_index(drop=True)
                        obj_2 = self.df_objects.loc[self.df_objects['ObjectID'] == positive_pair[1]].reset_index(drop=True)
                        if (not obj_1.empty) and (not obj_2.empty):
                            new_df = obj_1.join(obj_2, lsuffix="_1", rsuffix="_2")
                            new_df['PairInConstraint'] = 1
                            new_df['ModelID'] = model_id
                            self.my_df = pd.concat([self.my_df, new_df], axis=0)
                            positive_pairs_ctr += 1
            if positive_pairs_ctr > self.max_pos_pairs_ctr:
                self.max_pos_pairs_ctr = positive_pairs_ctr

            # IF POSITIVE RECORDS ADDED, CREATE ALSO NEGATIVE RECORDS
            if positive_pairs_ctr > 0:
                all_possible_pairs = set()
                all_combinations = itertools.combinations(sorted(all_obj_ids_in_model), 2)
                for possible_pair in all_combinations:
                    all_possible_pairs.add(possible_pair)
                combinations = positive_combinations ^ all_possible_pairs
                comb_set = set(combinations)
                comb_set = list(comb_set)
                random.shuffle(comb_set)
                if(len(comb_set) < positive_pairs_ctr):
                    self.my_df = self.my_df[self.my_df['ModelID_1'] != model_id]
                    continue
                for negative_subset in comb_set:
                    # STOP WHEN POSITIVE RECORDS NUM = NEGATIVE RECORDS NUM
                    # if negative_pairs_ctr == positive_pairs_ctr:
                    if negative_pairs_ctr == self.max_pos_pairs_ctr:
                        break
                    obj_1 = self.df_objects.loc[self.df_objects['ObjectID'] == negative_subset[0]].reset_index(drop=True)
                    obj_2 = self.df_objects.loc[self.df_objects['ObjectID'] == negative_subset[1]].reset_index(drop=True)
                    new_df = obj_1.join(obj_2, lsuffix="_1", rsuffix="_2")
                    new_df['PairInConstraint'] = 0
                    new_df['ModelID'] = model_id
                    if (not obj_1.empty) and (not obj_2.empty):
                        self.my_df = pd.concat([self.my_df, new_df], axis=0)
                        negative_pairs_ctr += 1

        # self.my_df.to_csv("pairs_un_balanced.csv", index=False)

        # self.my_df.drop = self.my_df.drop("ModelID_1",axis=1)
        # self.my_df.drop = self.my_df.drop("ModelID_2",axis=1)

        print("Unique models number in pairs dataframe", len(self.my_df['ModelID'].unique()))
        print("*" * 50)
        print("Max pos pairs is", self.max_pos_pairs_ctr)
        return self.my_df

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